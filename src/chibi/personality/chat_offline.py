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
("пирвіт" -> "привіт", "дякую" -> "дкяую") still match.

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
    "bye": "Бувай! Я буду тут.",
    "time": "Зараз {time}.",
    "quiet_on": "Добре, мовчу.",
    "quiet_off": "Я знову можу говорити!",
    "feed": "Ням! Дякую!",
    "pet": "Мур-р! Дякую!",
    "rest": "Добре, трішки відпочину.",
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


# Priority order: the first intent with a matching pattern wins.
INTENTS: tuple[Intent, ...] = (
    Intent("quiet_off", _rx(
        r"(?<!не )\b(можеш|можна|можете)( вже| знову| тепер)? (говорити|розмовляти|балакати)\b",
        r"(?<!не )\bговори\b(?! тихіше)",
        r"\bвже можна\b",
        r"\bповертайся\b",
        r"\bкінець тиші\b",
        r"\b(досить|годі) мовчати\b",
        r"\bвимкни (тишу|режим тиші)\b",
    )),
    Intent("quiet_on", _rx(
        r"\bпомовч\w*",
        r"\bзамовкн\w*",
        r"\bзамовч\w*",
        r"\bмовчи\b",
        r"\bтих(о|іше|енько)\b",
        r"\bне заважай\w*",
        r"\bтиш[аіу]\b",
        r"\bдай(те)? (мені )?спок(ій|ою)\b",
        r"\bцить(те)?\b",
        r"\bне (говори|говорити|балакай|балакати|розмовляй|розмовляти|базікай)\b",
        r"\bзаткни\w*",
        r"\bстули (пельку|писок|рота)\b",
        r"\bвідчепи\w*",
    ), ("помовч", "помовчи", "замовкни", "тихіше")),
    Intent("rude", _rx(
        r"(?<!не )\b(дурепа|дурень|дурна|тупа|тупий|тупиця|ідіот|ідіотка|дебіл|дебілка|кретин|кретинка"
        r"|придурок|придурка|бовдур|нікчема|потвора|лох|лошара)\b",
        r"\bти (погана|жахлива|огидна|бридка|нікчемна|бісиш|дратуєш|набридла)\b",
        r"\bненавиджу тебе\b",
        r"\bвідстій\w*",
    )),
    Intent("feed", _rx(
        r"\b(поїж|поїсти|їсти|їж|їжа|їжу|годую|погодую|підкріпись|пригощаю|пригощайся)\b",
        r"\bнагоду\w*",
        r"\bсмаколик\w*",
        r"\bперекус\w*",
        r"\b(печиво|печивко|печеньк\w*|цукерк\w*|тортик\w*)\b",
    ), ("нагодую", "смаколик", "печиво", "поїсти")),
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
    ), ("відпочинь",)),
    Intent("time", _rx(
        r"\bкотр(а|ій) (зараз )?годин\w*",
        r"\bскільки (зараз )?(часу|годин)\b",
        r"^котра\b",
        r"\bякий (зараз )?час\b",
        r"\bчас зараз\b",
    )),
    Intent("mood", _rx(
        r"\bпочуваєш\w*",
        r"\bнастр(ій|ою|оєчок)\b",
        r"\bсамопочуття\b",
        r"\bщо (ти )?відчуваєш\b",
    ), ("почуваєшся", "настрій")),
    Intent("name", _rx(
        r"\bяк (тебе )?(звати|звуть|кличуть|називати)\b",
        r"\bхто ти\b",
        r"\bти хто\b",
        r"\bтвоє ім'?я\b",
        r"\bім'?я твоє\b",
    )),
    Intent("capabilities", _rx(
        r"\b(що|шо) (ти )?(вмієш|умієш|можеш|знаєш)\b",
        r"\bдопомог\w*",
        r"\bдопоможи\b",
        r"\bhelp\b",
        r"\bкоманд\w*",
        r"\bможливост\w*",
        r"\bщо з тобою (робити|можна)\b",
        r"\bінструкці\w*",
    ), ("допомога", "допоможи", "команди")),
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
        r"\bяк (у тебе |в тебе )?(справи|справа|справці|ся маєш|поживаєш|життя|ти|воно|день)\b",
        r"\bщо нового\b",
        r"\bшо (як|нового)\b",
    )),
    Intent("thanks", _rx(
        r"\b(дякую|дякс|дяк|дяки|дякуємо|спасибі|спасибо|спс|сенкс|мерсі|thanks|thx)\b",
        r"\bthank you\b",
        r"\bвдячн\w*",
    ), ("дякую", "спасибі")),
    Intent("compliment", _rx(
        r"(?<!не )\b(класна|класний|класно|мила|миленька|милота|мило|гарна|гарненька|красива|красуня"
        r"|найкраща|найкращий|чудова|чудовий|чудово|крута|крутий|круто|розумна|розумниця|розумничка|молодець"
        r"|молодчинка|прекрасна|прекрасний|симпатична|сонечко|зайчик|котик|кицю|киця|лапочка|няшка|няша|супер)\b",
        r"\b(люблю|обожнюю) тебе\b",
        r"\bподобаєшся\b",
    ), ("молодець", "красива", "чудова", "найкраща", "розумниця")),
    Intent("bye", _rx(
        r"\b(бувай|бувайте|бувайчики|па па|пока|бай|bye|чао|прощавай|до зустрічі|до побачення|до завтра"
        r"|до скорого|на добраніч|добраніч|доброї ночі|на все добре|йду спати|я пішов|я пішла|мені пора)\b",
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


def detect_intent(text: str) -> str:
    """The intent name for a chat message ("unknown" if nothing matched)."""
    prepared = _prepare(text)
    if not prepared:
        return "unknown"
    for intent in INTENTS:
        if any(pattern.search(prepared) for pattern in intent.patterns):
            return intent.name
    words = [w for w in prepared.split() if len(w) >= MIN_FUZZY_LENGTH]
    if words:
        for intent in INTENTS:
            for trigger in intent.fuzzy_words:
                if any(_within_one_edit(word, trigger) for word in words):
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
        if isinstance(outcome, str) and outcome:
            keys = (f"interaction.{perform_name}.{outcome}",)
        return self._reply(intent, keys, action=reply_action)

    def _reply(self, intent: str, keys: tuple[str, ...], action: str = "", **variables: Any) -> ChatReply:
        speech = self._selector.say(*keys, solicited=True, **variables) if keys else None
        if speech is not None:
            return ChatReply(text=speech.text, state=speech.state, portrait=speech.portrait, action=action)
        fallback = FALLBACKS.get(intent, FALLBACKS["unknown"])
        values = {"name": self._settings().character.name, "time": variables.get("time") or self._time()}
        return ChatReply(text=fallback.format_map(values), action=action)

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
