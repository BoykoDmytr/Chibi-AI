"""Offline chat: an honest keyword responder for Phase 1 (no language model).

She recognises a handful of intents in Ukrainian (greetings, "як справи",
"помовч", feeding, petting, the time, ...) and answers with solicited lines
from the phrase bank.  Anything else gets an honest "I only understand simple
phrases until my AI brain arrives" - she never pretends to understand.

Matching works on :func:`~chibi.personality.text_uk.normalize`-d text with
word-boundary regexes, checked in a fixed priority order (being asked to be
quiet beats everything; an action like feeding beats a greeting in the same
message).  Cheap typo tolerance: letters stretched for emphasis are squeezed
("привіііт"), and single words within one edit of a known trigger word
("пирвіт" -> "привіт", "дякую" -> "дкяую") are corrected before the intent's
patterns run again, so a typo never bypasses the context rules below.

Who a message is about matters more than which words it contains: "не мовчи"
asks her to talk, "я хочу їсти" is about the user's lunch, "цей бос тупий" is
not an insult to her and "у мене поганий настрій" is not a question about hers.
So insults and compliments need to be addressed to her ("ти ...", or the whole
message is the word), actions skip first-person statements, and the user's
own feelings get their own small, supportive replies.

Privacy: the user's text is never logged - not even at debug level.  Only the
detected intent name is.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chibi.core.clock import Clock
from chibi.core.config import Settings
from chibi.personality.phrases import PhraseSelector
from chibi.personality.reactions import ReactionEngine
from chibi.personality.text_uk import format_clock, normalize
from chibi.settings.facade import ChatReply
from chibi.tamagotchi.needs import STAT_NAMES, Needs, NeedsModel

log = logging.getLogger(__name__)

#: NeedsModel.mood() word -> chat.mood.<suffix>
MOOD_KEYS: dict[str, str] = {
    "Сонна": "sleepy",
    "Голодна": "hungry",
    "Нудьгує": "bored",
    "Сумна": "sad",
    "Щаслива": "happy",
    "Весела": "cheerful",
    "Спокійна": "calm",
}

#: intent -> (perform() action, ChatReply.action)
ACTIONS: dict[str, tuple[str, str]] = {
    "feed": ("feed", "fed"),
    "pet": ("pet", "petted"),
    "rest": ("rest", "rest"),
}

#: perform() outcomes that mean the action really happened.  Only these set
#: ``ChatReply.action``: the chat prints "she ate / went to rest" from it, and a
#: refusal ("full", "too_much", "not_tired") must not claim that.
SUCCESS_OUTCOMES: dict[str, frozenset[str]] = {
    "feed": frozenset({"fed", "starving_fed"}),
    "pet": frozenset({"happy"}),
    "rest": frozenset({"sleep"}),
}

#: intent -> phrase keys (first usable wins)
INTENT_KEYS: dict[str, tuple[str, ...]] = {
    "greeting": ("chat.greeting",),
    "how_are_you": ("chat.how_are_you",),
    "name": ("chat.name",),
    "capabilities": ("chat.capabilities",),
    "joke": ("idle.joke",),
    "thought": ("idle.thought",),
    "question": ("idle.question",),
    "thanks": ("chat.thanks",),
    "compliment": ("chat.compliment",),
    "rude": ("chat.rude",),
    "sad": ("chat.user_sad",),
    "good": ("chat.user_good",),
    "bye": ("chat.bye",),
    "time": ("chat.time",),
    "quiet_on": ("quiet.ack",),
    "quiet_off": ("quiet.end",),
    "unknown": ("chat.unknown",),
}

#: Built-in replies for when the phrase bank has nothing usable (broken or
#: missing file).  ``{name}`` and ``{time}`` are filled in.
FALLBACKS: dict[str, str] = {
    "greeting": "Привіт! Я тут.",
    "how_are_you": "У мене все добре, дякую!",
    "mood": "Почуваюся непогано, дякую, що питаєш!",
    "name": "Мене звати {name}.",
    "capabilities": (
        "Поки я працюю без ШІ й розумію лише прості фрази: «привіт», «жарт», «котра година», «помовч»."
    ),
    "joke": "Ой, жарти закінчилися... Спитай мене трохи згодом!",
    "thought": "Я саме задумалася... і забула, про що. Буває!",
    "question": "Розкажи мені щось про себе?",
    "thanks": "Будь ласка!",
    "compliment": "Дякую, мені дуже приємно!",
    "rude": "Гаразд, я не ображаюся. Поговорімо про щось приємніше?",
    "sad": "Мені шкода. Я поруч.",
    "good": "Рада це чути!",
    "bye": "Бувай! Я буду тут.",
    "time": "Зараз {time}.",
    "quiet_on": "Добре, мовчу.",
    "quiet_off": "Я знову можу говорити!",
    "feed": "Ням! Дякую!",
    "pet": "Мур-р! Дякую!",
    "rest": "Добре, трішки відпочину.",
    # "<intent>.<outcome>" for refusals: the success line would claim she ate.
    "feed.full": "Дякую, але я вже наїлася!",
    "pet.too_much": "Дякую, але мені поки досить ласки.",
    "rest.not_tired": "Я ще зовсім не хочу спати!",
    "action_unavailable": "Ой, звідси я поки не можу цього зробити. Спробуй через меню!",
    "unknown": "Вибач, я поки розумію лише прості фрази: без ШІ я ще не дуже кмітлива.",
}


def _rx(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


@dataclass(frozen=True)
class Intent:
    name: str
    patterns: tuple[re.Pattern[str], ...]
    #: Single trigger words (>= 5 letters) matched with one typo allowed.
    fuzzy_words: tuple[str, ...] = ()
    #: The intent does not apply when one of these matches: "я хочу їсти" is
    #: about the user's lunch, not a request to feed her.
    exclude: tuple[re.Pattern[str], ...] = ()

    def matches(self, text: str) -> bool:
        if not any(pattern.search(text) for pattern in self.patterns):
            return False
        return not any(pattern.search(text) for pattern in self.exclude)


# Building blocks.  Input is normalize()-d: lower case, no punctuation, so a
# question mark cannot be checked and "$" means "end of the message".

# Words that may sit between "ти" and an adjective without changing whom it is
# about ("ти така мила", "ти в мене найкраща").  "не" is deliberately absent:
# "ти не тупа" is neither an insult nor a compliment.
_TY = (r"\bти( (така|такий|таке|просто|реально|дуже|ж|ну|справжня|справжній|повна|повний|ще|й|і|вже"
       r"|якась|якийсь|сьогодні|теж|також|завжди|моя|мій|моє|в мене|у мене|у нас|в нас))*")
# The same for the user speaking about themselves ("мені сьогодні якось сумно").
_ME = (r"\b(мені|мене|я|у мене|в мене)( (дуже|так|трохи|трішки|якось|сьогодні|зараз|зовсім|щось|аж|вже"
       r"|теж|також|просто|реально|настільки|геть|ну|ж|страшенно|жахливо|така|такий|якийсь|якась|знову"
       r"|постійно|все|усе))*")
_INSULTS = (r"дурепа|дурень|дурна|дурний|тупа|тупий|тупиця|ідіот|ідіотка|дебіл|дебілка|кретин|кретинка"
            r"|придурок|придурка|бовдур|нікчема|потвора|лох|лошара|відстій")
# Alone, only feminine or neutral forms are about her ("Ідіот!" is the teammate).
_INSULTS_HER = r"дурепа|дурна|тупа|тупиця|ідіотка|дебілка|кретинка|придурка|нікчема|потвора|лошара"
_SELF_INSULTS = (r"дурень|дурна|дурепа|дурний|тупий|тупа|тупиця|ідіот|ідіотка|бовдур|нікчема|невдаха|лох"
                 r"|лошара")
_COMPLIMENTS = (r"класна|класний|класно|мила|миленька|милий|гарна|гарненька|гарний|красива|красивий|красуня"
                r"|найкраща|найкращий|чудова|чудовий|чудово|крута|крутий|круто|розумна|розумний|розумниця"
                r"|розумничка|молодець|молодчинка|прекрасна|прекрасний|симпатична|сонечко|зайчик|котик|кицю"
                r"|киця|лапочка|няшка|няша|супер|золото|золотце|чарівна|неймовірна|найрідніша|солоденька")
# Alone, a compliment must be a feminine adjective or a pet name: "Класно!" or
# "Супер" answers "як справи?", "Гарна погода" is about the weather.
_COMPLIMENTS_ALONE = (r"класна|мила|миленька|гарна|гарненька|красива|красуня|найкраща|чудова|крута|розумна"
                      r"|розумниця|розумничка|молодець|молодчинка|прекрасна|симпатична|сонечко|зайчик|котик"
                      r"|кицю|киця|лапочка|няшка|няша|золото|золотце|чарівна|неймовірна|найрідніша|солоденька")
_SAD = (r"сумно|сумненько|погано|кепсько|паршиво|самотньо|зле|тоскно|тужно|важко|тяжко|боляче|страшно"
        r"|тривожно|гірко|прикро|сумна|сумний|самотня|самотній|засмучена|засмучений|пригнічена|пригнічений"
        r"|втомилася|втомився|втомлена|втомлений|виснажена|виснажений|депресія|стрес"
        r"|(поганий|кепський|паршивий|жахливий|сумний|зіпсований) настрій")
# "Все добре" is good news, a bare "Добре" is just "OK": weak words need a subject.
_GOOD_STRONG = r"чудово|класно|супер|круто|відмінно|прекрасно|чудесно|файно|кайф|кайфово|ідеально|топ"
_GOOD_WEAK = r"добре|нормально|норм|непогано|весело|окей|ок|гарно"
_NOT_ABOUT_HER = r"(?!.*\b(ти|тобі|тебе|тобою|твій|твоя|твоє|хочеш|будеш)\b)"
# An offer of food to her: "Поїж, а я теж поїм" is still feeding her.
_NO_FOOD_FOR_HER = r"(?!.*\b(ти|тобі|тебе|тобою|хочеш|будеш|поїж|їж|підкріпись|пригощайся|тримай|скуштуй)\b)"
_EATING = (r"їсти|поїсти|їм|з'їм|поїм|їв|їла|поїв|поїла|з'їв|з'їла|голодн\w*|зголодні\w*|перекушу|обідаю"
           r"|вечеряю|снідаю|пообідати|повечеряти|поснідати")
_DEATH = r"помер|померла|померли|загинув|загинула|загинули"
_MY = r"мій|моя|моє|мої|наш|наша|наше|наші"


# Priority order: the first intent with a matching pattern wins.
INTENTS: tuple[Intent, ...] = (
    Intent("quiet_off", _rx(
        # Whole-message requests only: "Ти можеш говорити українською?" is a question.
        r"^((ну|все|усе|добре|гаразд|окей|тепер) )*(ти )?(вже |знову |тепер )?(можеш|можна|можете)"
        r"( вже| знову| тепер)? (говорити|розмовляти|балакати)( (вже|знову|тепер|будь ласка|зі мною))*$",
        r"(?<!не )\b(по)?говори\b(?! (тихіше|тихо|тихенько|менше|поменше))",
        r"^вже можна$",
        r"\bповертайся\b",
        r"\bкінець (тиші|мовчанки)\b",
        r"\b(досить|годі|хватить) (мовчати|мовчанки)\b",
        r"\b(не|не треба|не потрібно|не варто) (мовчи|мовчати|мовчіть)\b",
        # Cancelling quiet mode ("скасуй режим тиші", "тиша набридла", "мені тиша не
        # потрібна"), with the words next to each other: "досить говорити, тиша!" and
        # "тиша, не треба нічого казати" ask for quiet.
        r"\b(вимкн\w*|скасу\w*|прибер\w*|зніми|зняти|відключ\w*)( (будь ласка|вже|нарешті|цей|цю|свій|свою"
        r"|мені|режим))* (тиш\w*|мовчанк\w*)",
        r"\b(не треба|не потрібно|досить|годі|хватить|кінець) (тиш\w*|мовчанк\w*)\b",
        r"\b(тиш\w*|мовчанк\w*)( \w+){0,2} набридл\w*",
        r"\b(тиш\w*|мовчанк\w*)( (мені|вже|зараз|більше))* (не треба|не потрібн\w*)$",
    )),
    Intent("quiet_on", _rx(
        r"\bпомовч\w*",
        r"\bзамовкн\w*",
        r"\bзамовч\w*",
        r"(?<!не )\bмовч(и|іть)\b",
        r"^мовчати$",
        r"\bтих(іше|енько)\b",
        # A bare "тихо" is a command only at the start: "Чому так тихо?" is not.
        r"^((а )?ну |будь ласка |ану |а тепер |все )?тихо\b(?! (тут|якось|сьогодні|стало|чомусь|як))",
        r"\b(сиди|посидь|поводься|побудь) (тихо|тихенько|тихіше)\b",
        r"\b(говори|розмовляй) (тихо|менше|поменше)\b",
        r"\b(досить|годі|хватить|перестань|припини) (говорити|балакати|базікати|теревенити|розмовляти|балачок"
        r"|базікання|теревенів)\b",
        r"^(тс+|ш( ш)*|ша)\b",
        r"\bтс+\b",
        r"\bш( ш)+\b",
        r"\bне заважай\w*",
        r"\bне відволікай\w*",
        r"\bтиш[аіу]\b",
        r"\bдай(те)? (мені )?спок(ій|ою)\b",
        r"\bдай(те)? (мені )?(поспати|попрацювати|зосередитися|зосередитись|подумати)\b",
        # "відстань" is also the noun "distance": only the imperative counts.
        r"^((ну|та|ой|будь ласка) )?відстан(ь|ьте)\b(?! (до|між|\d))",
        r"\bвідстан(ь|ьте) від мене\b",
        r"\bцить(те)?\b",
        # "Не говори так" / "не говори дурниць" is about what she says, not whether.
        r"\bне (говори|говорити|балакай|балакати|розмовляй|розмовляти|базікай)\b"
        r"(?! (так|такого|таке|дурниць)\b)",
        r"\bзаткни\w*",
        r"\bстули (пельку|писок|рота)\b",
        r"\bвідчепи\w*",
    ), ("помовч", "помовчи", "замовкни", "тихіше")),
    Intent("sad", _rx(
        _ME + rf" ({_SAD}|{_SELF_INSULTS})\b",
        rf"^{_NOT_ABOUT_HER}.*\b(поганий|кепський|паршивий|жахливий|зіпсований|сумний) настрій\b",
        rf"^{_NOT_ABOUT_HER}.*\bнастрій( (в мене|у мене|сьогодні|якийсь|зовсім))*"
        r" (поганий|кепський|паршивий|жахливий|на нулі|ніякий|зіпсований|сумний)\b",
        r"^((все|усе|якось|щось|ну) )*(погано|кепсько|паршиво|жахливо|сумно|сумненько|тоскно|так собі|зле"
        r"|не дуже)( а (ти|в тебе|у тебе)( як)?)?$",
        rf"\b(у|в) мене ({_DEATH})\b",
        rf"\b({_DEATH}) ({_MY})\b",
        rf"\b({_MY}) \w+ ({_DEATH})\b",
        r"\bне стало\b",
        r"\bплач(у|еться)\b",
        r"\bхочеться плакати\b",
    )),
    Intent("rude", _rx(
        _TY + rf" ({_INSULTS})\b",
        r"\bти (мене |мені )?(погана|жахлива|огидна|бридка|нікчемна|бісиш|дратуєш|набридла|дістала)\b",
        rf"^((ну|ох|ой|та) )?({_INSULTS_HER})( ти)?$",
        r"\bненавиджу тебе\b",
        r"\bтебе ненавиджу\b",
    )),
    Intent("feed", _rx(
        r"\b(поїж|поїсти|їсти|їж|їжа|їжу|годую|погодую|підкріпись|пригощаю|пригощайся)\b",
        r"\bнагоду\w*",
        r"\bсмаколик\w*",
        r"\bперекус\w*",
        r"\b(печиво|печивко|печеньк\w*|цукерк\w*|тортик\w*)\b",
    ), ("нагодую", "смаколик", "печиво", "поїсти"),
        # "Я хочу їсти", "мені треба поїсти", "вже можна їсти?": the user eats, not her.
        exclude=_rx(rf"^{_NO_FOOD_FOR_HER}.*\b(я|мені|ми|нам|можна)\b.*\b({_EATING})\b"),
    ),
    Intent("pet", _rx(
        r"\bпоглад\w*",
        r"\bгладж\w*",
        r"\bглад(ь|ити|имо)\b",
        r"\bобійм\w*",
        r"\bобнім\w*",
        r"\bпригорн\w*",
        r"\bпочуха\w*",
        r"\bцьом\w*",
    ), ("погладжу", "погладити", "обійми", "обіймашки")),
    Intent("rest", _rx(
        r"\b(йди|іди|піди|лягай|пора|час|марш) (спати|поспи|поспати|відпочивати|відпочити|баїньки|спатки"
        r"|в ліжко|в ліжечко)\b",
        r"\bвідпочинь\w*",
        r"\bпоспи\b",
        r"\b(спатки|баїньки)\b",
        r"\bхочеш спати\b",
    ), ("відпочинь",),
        # "Мені пора спати" is the user going to bed (a goodbye), not an order to her.
        exclude=_rx(r"\b(мені|нам) (вже |теж |також |саме )?(пора|час)\b"),
    ),
    Intent("time", _rx(
        r"\bкотр(а|ій) (зараз )?годин\w*",
        r"\bскільки (зараз )?(часу|годин)\b",
        r"^котра\b",
        r"\bякий (зараз )?час\b",
        r"\bчас зараз\b",
    )),
    Intent("mood", _rx(
        # Her mood only: "у мене поганий настрій" is the user's (see "sad").
        r"\bпочуваєш\w*",
        r"\b(твій|який|як|який у тебе|який в тебе) настр\w*",
        r"\b(у|в) тебе( \w+)? настр\w*",
        r"\bнастр\w* (у тебе|в тебе|твій)\b",
        r"\b(як|яке) (твоє |у тебе |в тебе )?самопочуття\b",
        r"\bтвоє самопочуття\b",
        r"\bщо (ти )?відчуваєш\b",
    ), ("почуваєшся",)),
    Intent("name", _rx(
        r"\bяк (тебе )?(звати|звуть|кличуть|називати)\b",
        r"\bхто ти\b",
        r"\bти хто\b",
        r"\bтвоє ім'?я\b",
        r"\bім'?я твоє\b",
    )),
    Intent("capabilities", _rx(
        r"\b(що|шо) (ти )?(вмієш|умієш|можеш|знаєш)\b",
        # "допомога", not "дякую за допомогу" (thanks) or "ти мені допомогла".
        r"\bдопомог(а|и)\b",
        r"\bдопоможи\w*",
        r"\bhelp\b",
        # "Моя команда знову злила катку" is about a game.
        r"\b(які|твої|список|всі|усі)( (в|у) тебе| є)? команд\w*",
        r"^команди$",
        r"\bможливост\w*",
        r"\bщо з тобою (робити|можна)\b",
        r"\bінструкці\w*",
    ), ("допомога", "допоможи")),
    Intent("joke", _rx(
        r"\bжарт\w*",
        r"\bпожартуй\w*",
        r"\bанекдот\w*",
        r"\bрозсміш\w*",
        r"\bсмішн\w*",
        r"\bприкол\w*",
        r"\bгумор\w*",
    ), ("анекдот", "пожартуй", "розсміши")),
    Intent("thought", _rx(
        r"\bрозкажи (мені )?(щось|що небудь|щонебудь)\b",
        r"\bпро що (ти )?(думаєш|мрієш)\b",
    )),
    Intent("question", _rx(
        r"\b(запитай|спитай|задай|постав) (мене|мені|питання|запитання|щось)\b",
    )),
    Intent("how_are_you", _rx(
        r"\bяк (у тебе |в тебе )?(справи|справа|справці|ся маєш|поживаєш|життя|воно|день)\b",
        # "як ти?" - but not "як ти думаєш, ...".
        r"\bяк ти( там| сьогодні| взагалі| поживаєш| загалом)?$",
        r"^(а |ну )?ти як( там| сьогодні)?$",
        # "Чудово, а в тебе?" answers her "як настрій?" and asks back.
        rf"\b({_GOOD_STRONG}|{_GOOD_WEAK})\b.*\bа (ти|в тебе|у тебе)( як)?$",
        r"\bщо нового\b",
        r"\bшо (як|нового)\b",
    )),
    Intent("good", _rx(
        # The user is fine ("Все супер", "У мене все добре"): the whole message, so
        # "все добре з тобою?" or "добре, помовч" never land here.
        rf"^(дякую )?(((у|в) мене|мені) )?((теж|також|все|усе|дуже|просто|сьогодні|цілком|загалом|ну|та) )*"
        rf"({_GOOD_STRONG})( (дякую|все|усе|у мене|в мене))*$",
        rf"^(дякую )?((((у|в) мене|мені) ((теж|також|все|усе|дуже|просто|сьогодні|цілком|загалом) )*)"
        rf"|(((теж|також|ну|та) )*(все|усе) ((теж|також|дуже|просто|цілком) )*))({_GOOD_WEAK})( дякую)?$",
    )),
    Intent("thanks", _rx(
        r"\b(дякую|дякс|дяк|дяки|дякуємо|спасибі|спасибо|спс|сенкс|мерсі|thanks|thx)\b",
        r"\bthank you\b",
        r"\bвдячн\w*",
    ), ("дякую", "спасибі")),
    Intent("compliment", _rx(
        _TY + rf" ({_COMPLIMENTS})\b",
        rf"^((ну|яка|ох|ой|о|моя|ти ж|ти моя|ти в мене) )*({_COMPLIMENTS_ALONE})"
        r"( (ти|моя|ти моя|ти в мене|в мене|у мене))*$",
        r"^((як|яка|ну|ой|ох) )*(ж )?(мило|милота|миленько)$",
        r"\b(люблю|обожнюю|кохаю) тебе\b",
        r"\bтебе (люблю|обожнюю|кохаю)\b",
        r"\bподобаєшся\b",
    ), ("молодець", "красива", "чудова", "найкраща", "розумниця")),
    Intent("bye", _rx(
        r"\b(бувай|бувайте|бувайчики|па па|пока|бай|bye|чао|прощавай|до зустрічі|до побачення|до завтра"
        r"|до скорого|на добраніч|добраніч|на все добре|йду спати|піду спати|я пішов|я пішла)\b",
        r"\b(мені|нам) (вже |теж |також |саме )?пора\b",
        r"\b(мені|нам) (вже |теж |також |саме )?час (йти|іти|спати|бігти)\b",
        r"\b(спокійної|доброї|гарної|солодкої|тихої) ночі\b",
        # Wishes for the rest of the day; "доброго дня/вечора" stays a greeting.
        r"\b(гарного|хорошого|чудового|вдалого|приємного|гарних|хороших|вдалих)"
        r" (дня|вечора|вихідних|тижня)\b",
        r"\bсолодких снів\b",
        r"^па$",
        r"^(ну )?поки( поки)?$",
    ), ("бувайте", "добраніч")),
    Intent("greeting", _rx(
        r"\b(привіт\w*|привт|прівіт|првіт|привет\w*|хелоу|хеллоу|хелло|хело|hello|hi|hey|вітаю|вітання"
        r"|вітаннячко|здоровенькі|здоровеньки|салют|добридень|добривечір)\b",
        r"\bздоров\b(?!')",
        r"^(хей|йо|йоу)\b",
        r"^(хай|ку)( всім| усім| тобі)?$",
        r"\b(доброго|добрий|добра|доброї) (ранку|ранок|дня|день|вечора|вечір|днини)\b",
    ), ("привіт", "привітик", "вітаю", "добридень", "добривечір")),
)

_STRETCHED = re.compile(r"(\w)\1{2,}")


def _prepare(text: str) -> str:
    """Normalise and squeeze letters stretched for emphasis ("привіііт" -> "привіт")."""
    return _STRETCHED.sub(r"\1", normalize(text))


def _within_one_edit(a: str, b: str) -> bool:
    """Optimal string alignment distance <= 1 (insert, delete, substitute or swap neighbours)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) != 2 or diffs[1] != diffs[0] + 1:
            return False
        i, j = diffs
        return a[i] == b[j] and a[j] == b[i]
    shorter, longer = (a, b) if la < lb else (b, a)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


MIN_FUZZY_LENGTH = 5


def _corrected(words: list[str], triggers: tuple[str, ...]) -> str | None:
    """The message with every word that is one typo away from a trigger replaced by
    that trigger, or None when nothing was corrected."""
    changed = False
    out: list[str] = []
    for word in words:
        if len(word) >= MIN_FUZZY_LENGTH and word not in triggers:
            fix = next((t for t in triggers if _within_one_edit(word, t)), None)
            if fix is not None:
                word, changed = fix, True
        out.append(word)
    return " ".join(out) if changed else None


def detect_intent(text: str) -> str:
    """The intent name for a chat message ("unknown" if nothing matched)."""
    prepared = _prepare(text)
    if not prepared:
        return "unknown"
    for intent in INTENTS:
        if intent.matches(prepared):
            return intent.name
    # Typos: correct towards an intent's trigger words, then apply that intent's
    # own patterns, so "ця гра чудвоа" is still not a compliment to her.
    words = prepared.split()
    for intent in INTENTS:
        if intent.fuzzy_words:
            corrected = _corrected(words, intent.fuzzy_words)
            if corrected is not None and intent.matches(corrected):
                return intent.name
    return "unknown"


class OfflineChatResponder:
    def __init__(
        self,
        selector: PhraseSelector,
        reactions: ReactionEngine,
        settings: Callable[[], Settings],
        clock: Clock,
        perform: Callable[[str], str] | None = None,
        needs: Callable[[], Mapping[str, float]] | None = None,
    ) -> None:
        """``perform("feed" | "pet" | "rest")`` runs the tamagotchi interaction
        (without its own announcement) and returns the outcome string;
        ``needs()`` returns the current needs (``Needs.to_dict()``-like, a
        ``"mood"`` entry is used as is) for "як ти себе почуваєш"."""
        self._selector = selector
        self._reactions = reactions
        self._settings = settings
        self._clock = clock
        self._perform = perform
        self._needs = needs
        self._model = NeedsModel()

    def respond(self, text: str) -> ChatReply:
        """Reply to one chat message.  Always returns a reply, never raises."""
        intent = "unknown"
        try:
            intent = detect_intent(text if isinstance(text, str) else "")
            log.debug("chat intent: %s", intent)
            return self._handle(intent)
        except Exception:
            log.exception("offline chat failed for intent %s", intent)
            return ChatReply(text=FALLBACKS["unknown"])

    # -- handlers ------------------------------------------------------------------------------

    def _handle(self, intent: str) -> ChatReply:
        if intent == "quiet_on":
            self._reactions.set_quiet(self._settings().behavior.quiet_minutes)
            return self._reply(intent, INTENT_KEYS[intent], action="quiet_on")
        if intent == "quiet_off":
            self._reactions.clear_quiet()
            return self._reply(intent, INTENT_KEYS[intent], action="quiet_off")
        if intent in ACTIONS:
            return self._action(intent)
        if intent == "time":
            return self._reply(intent, INTENT_KEYS[intent], time=self._time())
        if intent == "mood":
            mood = self._mood_key()
            keys = (f"chat.mood.{mood}", "chat.how_are_you") if mood else ("chat.how_are_you",)
            return self._reply(intent, keys)
        return self._reply(intent, INTENT_KEYS.get(intent, INTENT_KEYS["unknown"]))

    def _action(self, intent: str) -> ChatReply:
        perform_name, reply_action = ACTIONS[intent]
        if self._perform is None:
            return ChatReply(text=FALLBACKS["action_unavailable"])
        try:
            outcome = self._perform(perform_name)
        except Exception:
            log.exception("chat action %s failed", perform_name)
            return ChatReply(text=FALLBACKS["action_unavailable"])
        keys: tuple[str, ...] = ()
        fallback = intent
        if isinstance(outcome, str) and outcome:
            keys = (f"interaction.{perform_name}.{outcome}",)
            if f"{intent}.{outcome}" in FALLBACKS:
                fallback = f"{intent}.{outcome}"
        # A refusal (or an outcome we do not know) did not feed / pet / rest her.
        succeeded = isinstance(outcome, str) and outcome in SUCCESS_OUTCOMES[perform_name]
        return self._reply(intent, keys, action=reply_action if succeeded else "", fallback=fallback)

    def _reply(self, intent: str, keys: tuple[str, ...], action: str = "", fallback: str | None = None,
               **variables: Any) -> ChatReply:
        speech = self._selector.say(*keys, solicited=True, **variables) if keys else None
        if speech is not None:
            return ChatReply(text=speech.text, state=speech.state, portrait=speech.portrait, action=action)
        template = FALLBACKS.get(fallback or intent, FALLBACKS["unknown"])
        values = {"name": self._settings().character.name, "time": variables.get("time") or self._time()}
        return ChatReply(text=template.format_map(values), action=action)

    def _time(self) -> str:
        return format_clock(datetime.fromtimestamp(self._clock.now()))

    def _mood_key(self) -> str | None:
        if self._needs is None:
            return None
        try:
            snapshot: Any = self._needs()
        except Exception:
            log.exception("could not read needs for the chat")
            return None
        if isinstance(snapshot, Mapping):
            mood = snapshot.get("mood")
            data: Mapping[str, Any] = snapshot
        else:  # a view object such as settings.facade.NeedsView
            mood = getattr(snapshot, "mood", None)
            data = {name: getattr(snapshot, name) for name in STAT_NAMES if hasattr(snapshot, name)}
        if not isinstance(mood, str) or not mood.strip():
            mood = self._model.mood(Needs.from_dict(data))
        return MOOD_KEYS.get(mood.strip().capitalize())
