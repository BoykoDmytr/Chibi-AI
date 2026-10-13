"""Golden corpus for the offline chat: realistic Ukrainian messages -> expected intent.

One table instead of a pile of one-off regression tests, so a regex that fixes one
phrase and breaks another fails here at once.  It covers every message named in the
review findings UK-1..UK-8 / UK-14, the regression check REG-1..REG-9 and the blind
test BT (both the messages that must match and the ones that must not), plus typos,
MiXeD case, emoji, apostrophe variants, quoted words, roleplay actions, her name as a
vocative in any position ("Міє, ...", "..., Міє", "Мія ...") and a renamed character
("Кіра" -> "Кіро, будь тихо").

The second half runs messages end to end through OfflineChatResponder and checks the
side effects that matter: quiet mode on/off and whether perform("feed" / "pet" /
"rest") was called.
"""

import random
from datetime import datetime

import pytest

from chibi.core.config import Settings
from chibi.memory.repositories import PhraseHistoryRepository
from chibi.personality.chat_offline import OfflineChatResponder, address_forms, detect_intent
from chibi.personality.phrases import PhraseBank, PhraseSelector
from chibi.personality.reactions import ReactionEngine
from chibi.personality.text_uk import render

BANK = PhraseBank.load()

# (message, expected intent) with her default name, "Мія" (vocative "Міє").
CORPUS: list[tuple[str, str]] = [
    # -- greetings --------------------------------------------------------------------------
    ("Привіт!", "greeting"),
    ("ПРИВІТ", "greeting"),
    ("привііііт 👋", "greeting"),
    ("пирвіт", "greeting"),
    ("привiт", "greeting"),  # Latin "i"
    ("Привіт, Міє!", "greeting"),
    ("Міє, привіт", "greeting"),
    ("Привіт, сонечко", "greeting"),
    ("Доброго ранку, сонечко ☀️", "greeting"),
    ("Добрий вечір, Міє", "greeting"),
    ("Хай, Міє", "greeting"),
    ("Привіт, мила", "greeting"),
    # -- how are you (REG-4) ----------------------------------------------------------------
    ("Як справи?", "how_are_you"),
    ("Як справи, Міє?", "how_are_you"),
    ("Міє, як справи?", "how_are_you"),
    ("Привіт, Міє, як справи?", "how_are_you"),
    ("Як ти?", "how_are_you"),
    ("Як ти, Міє?", "how_are_you"),
    ("Як ти, Мія?", "how_are_you"),
    ("Як ти там, Міє?", "how_are_you"),
    ("Міє, як ти?", "how_are_you"),
    ("Як ти там поживаєш?", "how_are_you"),
    ("Як ти себе маєш?", "how_are_you"),
    ("Як ти, все добре?", "how_are_you"),
    ("Як ти, нормально?", "how_are_you"),
    ("Як ти, сонечко?", "how_are_you"),
    ("Чудово, а в тебе?", "how_are_you"),  # UK-5: answers her "як настрій?" and asks back
    ("Добре, а ти як?", "how_are_you"),
    ("У мене гарний настрій, а в тебе?", "how_are_you"),
    ("Як ти думаєш, піца з ананасами норм?", "unknown"),  # UK-8
    ("Як ти думаєш, Міє?", "unknown"),
    # -- her mood (REG-4, UK-5) -------------------------------------------------------------
    ("Настрій?", "mood"),
    ("А настрій?", "mood"),
    ("Настрій як?", "mood"),
    ("Ну що, настрій?", "mood"),
    ("Міє, настрій?", "mood"),
    ("Самопочуття?", "mood"),
    ("Як твій настрій?", "mood"),
    ("Як настрій?", "mood"),
    ("Як ти себе почуваєш, Міє?", "mood"),
    # -- the user's feelings: support (UK-5, UK-6, UK-14) -----------------------------------
    ("Мені сумно", "sad"),
    ("мені сумно 😔", "sad"),
    ("Мені погано", "sad"),
    ("Мені самотньо", "sad"),
    ("Все погано", "sad"),
    ("Не дуже, а ти?", "sad"),
    ("Я втомився", "sad"),
    ("У мене поганий настрій", "sad"),
    ("Я такий дурень, знову забув зберегтися", "sad"),
    ("У мене помер кіт", "sad"),
    ("Мій кіт помер", "sad"),
    ("Помер мій кіт", "sad"),
    ("Бабуся померла 😢", "sad"),
    ("Привіт! Я втомився", "sad"),
    # A hug or help asked for in a sad message is comfort, not a command: support wins.
    ("Мені погано, обійми мене", "sad"),
    ("Я сумний, погладь мене", "sad"),
    ("Мені погано, допоможи", "sad"),
    ("Мені сумно, нагодуй мене", "sad"),
    ("Мені сумно, поговори зі мною", "sad"),
    # -- a feeling plus an explicit command: the command wins (REG-1) ----------------------
    ("Я втомився, йди спати", "rest"),
    ("Я теж втомився, йди спати", "rest"),
    ("Мені сумно, розкажи жарт", "joke"),
    ("Мені сумно, пожартуй", "joke"),
    ("Мені сумно, розкажи щось", "thought"),
    ("Мені самотньо, спитай мене щось", "question"),
    ("Я втомився, на добраніч", "bye"),
    ("Я втомилася, на добраніч", "bye"),
    ("Я втомився, бувай", "bye"),
    ("Я втомився, піду спати", "bye"),
    ("Я втомився, котра година?", "time"),
    ("Мені погано, помовч", "quiet_on"),
    # -- not grief, not self-criticism (REG-9) ----------------------------------------------
    ("Мій персонаж помер", "unknown"),
    ("Мій перс загинув", "unknown"),
    ("Загинув мій тіммейт", "unknown"),
    ("Наш танк загинув", "unknown"),
    ("Я помер на босі", "unknown"),
    ("Мені тупий тіммейт попався", "unknown"),
    ("Мені дурна гра попалася", "unknown"),
    ("Мені не сумно", "unknown"),
    # -- the user is fine (UK-5) ------------------------------------------------------------
    ("Все супер", "good"),
    ("Супер!", "good"),
    ("У мене все класно", "good"),
    ("Все добре, дякую", "good"),
    ("Все супер, Міє", "good"),
    ("У мене гарний настрій", "good"),
    ("Настрій чудовий", "good"),
    ("Добре", "unknown"),
    ("Гарна погода сьогодні", "unknown"),
    ("Ця гра чудова", "unknown"),
    # -- compliments (REG-5, UK-8) ----------------------------------------------------------
    ("Ти класна", "compliment"),
    ("Ти така мила", "compliment"),
    ("Ти найкраща ❤️", "compliment"),
    ("МоЛоДеЦь", "compliment"),
    ("Сонечко", "compliment"),
    ("Ти моє сонечко", "compliment"),
    ("Ти моя хороша", "compliment"),
    ("Ти золотко", "compliment"),
    ("Молодець, Міє!", "compliment"),
    ("Молодець, Міа", "compliment"),
    ("Мія молодець", "compliment"),
    ("Розумниця, Міє", "compliment"),
    ("Розумниця моя, Міє", "compliment"),
    ("Міє, ти молодець", "compliment"),
    ("Ти молодець, Міє", "compliment"),
    ("Молодець, що нагадала", "compliment"),
    ("Молодець, дівчинко", "compliment"),
    ("Як мило з твого боку", "compliment"),
    ("Це мило з твого боку", "compliment"),
    ("Це так мило", "compliment"),
    ("Дуже мило", "compliment"),
    ("Так мило", "compliment"),
    ("Ти неймовірно мила", "compliment"),
    ("Ти надзвичайно мила", "compliment"),
    ("Ти виглядаєш класно", "compliment"),
    ("Я тебе люблю", "compliment"),
    ("Я тебе люблю, Міє", "compliment"),
    ("ти чудвоа", "compliment"),  # typo
    # -- rude, only when it is about her (REG-7, UK-6) --------------------------------------
    ("Ти тупа", "rude"),
    ("Ти що, дурна?", "rude"),
    ("Ти що тупа", "rude"),
    ("Ти зовсім дурна?", "rude"),
    ("Ти геть тупа", "rude"),
    ("Ти взагалі тупа", "rude"),
    ("Міє, ти що, дурна?", "rude"),
    ("Я тебе ненавиджу", "rude"),
    ("ти ж не дурна", "unknown"),
    ("Ти зовсім не дурна", "unknown"),
    ("Цей бос тупий", "unknown"),
    ("Ця гра відстій", "unknown"),
    ("Ідіот!", "unknown"),
    # -- be quiet (REG-3, UK-2) -------------------------------------------------------------
    ("Помовч", "quiet_on"),
    ("ПОМОВЧ", "quiet_on"),
    ("помвоч", "quiet_on"),  # typo
    ("Міє, помовч", "quiet_on"),
    ("Помовч, Міє", "quiet_on"),
    ("Сонечко, помовч", "quiet_on"),
    ("Будь тихо", "quiet_on"),
    ("Міє, будь тихо", "quiet_on"),
    ("Міє, тихо", "quiet_on"),
    ("Тихо, Міє", "quiet_on"),
    ("Ага, тихо", "quiet_on"),
    ("Давай тихо", "quiet_on"),
    ("Тепер тихо", "quiet_on"),
    ("Можна тихо?", "quiet_on"),
    ("Будь тихою", "quiet_on"),
    ("Тихiше", "quiet_on"),  # Latin "i"
    ("тсс", "quiet_on"),
    ("Тссс! 🤫", "quiet_on"),
    ("Ш-ш-ш", "quiet_on"),
    ("Ша!", "quiet_on"),
    ("Тсс, я на дзвінку", "quiet_on"),
    ("Відстань", "quiet_on"),
    ("Відстань від мене", "quiet_on"),
    ("Відвали", "quiet_on"),
    ("Не відволікай мене, я працюю", "quiet_on"),
    ("Дай поспати", "quiet_on"),
    ("Помовчи, дурна", "quiet_on"),
    ("Ти можеш говорити тихіше?", "quiet_on"),
    # ...and what only looks like it (UK-1, REG-9).
    ("Чому так тихо?", "unknown"),
    ("Тихо тут якось", "unknown"),
    ("Тихо тут сьогодні", "unknown"),
    ("Тепер тут тихо", "unknown"),
    ("Я тихенько граю", "unknown"),
    ("Хм, дай подумати", "unknown"),
    ("Зараз, дай подумати", "unknown"),
    ("Дай подумати", "unknown"),
    ("Не відволікайся", "unknown"),
    ("Не зараз", "unknown"),
    ("Яка відстань до Місяця?", "unknown"),
    ("Не говори так!", "unknown"),
    # -- talk again (REG-2, UK-1) -----------------------------------------------------------
    ("Не мовчи!", "quiet_off"),
    ("Не мовчи, Міє", "quiet_off"),
    ("Не треба тиші", "quiet_off"),
    ("Вимкнути тишу", "quiet_off"),
    ("Скасуй режим тиші", "quiet_off"),
    ("Досить тиші", "quiet_off"),
    ("Мені тиша не потрібна", "quiet_off"),
    ("Тиша набридла", "quiet_off"),
    ("Можеш говорити", "quiet_off"),
    ("Можеш говорити, Міє", "quiet_off"),
    ("Міє, можеш говорити", "quiet_off"),
    ("Міє, можеш вже говорити", "quiet_off"),
    ("Мія, можна говорити", "quiet_off"),
    ("Можеш говорити далі", "quiet_off"),
    ("Можеш говорити, я закінчив", "quiet_off"),
    ("Можна говорити, я вільний", "quiet_off"),
    ("Можеш говорити, дякую", "quiet_off"),
    ("Ти вже можеш говорити, дякую", "quiet_off"),
    ("Можеш говорити, я повернувся", "quiet_off"),
    ("Дякую, можеш говорити", "quiet_off"),
    ("Я закінчив, можеш говорити", "quiet_off"),
    ("Поговори зі мною", "quiet_off"),
    # ...but a question about how she talks is not a request (UK-7).
    ("Ти можеш говорити українською?", "unknown"),
    ("Ти можеш розмовляти англійською?", "unknown"),
    ("Можеш говорити повільніше?", "unknown"),
    ("Вже можна їсти?", "unknown"),
    # -- feeding: offers to her feed her, the user's own hunger does not (REG-8, UK-8) ------
    ("Я приніс поїсти", "feed"),
    ("Я принесла поїсти", "feed"),
    ("Я дам їсти", "feed"),
    ("Я приготувала поїсти", "feed"),
    ("Тримай печиво, Міє", "feed"),
    ("Міє, хочеш їсти?", "feed"),
    ("Я тебе нагодую", "feed"),
    ("З’їж печиво", "feed"),  # right single quote
    ("Поїж, а я теж поїм", "feed"),
    ("Я хочу їсти", "unknown"),
    ("Я голодна, піду поїсти", "unknown"),
    ("Я голодний", "unknown"),
    ("Мені треба поїсти", "unknown"),
    ("Я йду їсти", "unknown"),
    ("Я з’їв печиво", "unknown"),
    ("Я їм печиво", "unknown"),
    ("Хочу їсти", "unknown"),
    ("Їсти хочу", "unknown"),
    ("Мій кіт хоче їсти", "unknown"),
    ("Нагодуй мене", "unknown"),  # the user wants food (and the typo pass must not "fix" that)
    ("Хочу дати тобі поїсти", "feed"),
    # -- petting ----------------------------------------------------------------------------
    ("Погладжу тебе, Міє", "pet"),
    ("Обійми мене", "pet"),
    ("погаджу", "pet"),  # typo
    # -- sleep: an order to her vs the user going to bed (REG-8) ----------------------------
    ("Йди спати, мені теж пора", "rest"),
    ("Мені пора, йди спати", "rest"),
    ("Нам пора спати", "rest"),
    ("Міє, йди спати", "rest"),
    ("Йди спати, а я ще пограю", "rest"),
    ("Тобі треба відпочити", "rest"),
    ("Мені пора, а тобі спати", "rest"),
    ("Мені пора спати", "bye"),
    ("Мені теж пора спати", "bye"),
    # -- time, name, capabilities (REG-6, UK-8) ---------------------------------------------
    ("Котра година?", "time"),
    ("Міє, котра година?", "time"),
    ("Як тебе звати?", "name"),
    ("як твоє ім’я", "name"),  # right single quote
    ("як твоє імʼя", "name"),  # modifier letter apostrophe
    ("як твоє ім`я", "name"),  # backtick
    ("як твоє імя", "name"),  # no apostrophe at all
    ("Що ти вмієш?", "capabilities"),
    ("Покажи команди", "capabilities"),
    ("Підкажи команди", "capabilities"),
    ("Команди покажи", "capabilities"),
    ("Що за команди?", "capabilities"),
    ("Напиши команди", "capabilities"),
    ("Розкажи команди", "capabilities"),
    ("Які ти знаєш команди?", "capabilities"),
    ("Які в тебе команди?", "capabilities"),
    ("кманди", "capabilities"),  # typo
    ("кманди будь ласка", "capabilities"),
    ("Моя команда знову злила катку", "unknown"),
    ("Наша команда виграла!", "unknown"),
    # -- jokes, thanks, goodbyes ------------------------------------------------------------
    ("Розкажи жарт, Міє", "joke"),
    ("Міє, розкажи анекдот", "joke"),
    ("анекодт", "joke"),  # typo
    ("Дякую!", "thanks"),
    ("ДЯКУЮ!!!", "thanks"),
    ("дкяую", "thanks"),  # typo
    ("Дякую, мала", "thanks"),
    ("Дякую за допомогу", "thanks"),
    ("Спокійної ночі", "bye"),
    ("Гарного дня!", "bye"),
    ("Добраніч, люба", "bye"),
    ("Поки, Міє", "bye"),
    ("Бувай, Міє 👋", "bye"),
    # -- nothing she can honestly answer ----------------------------------------------------
    ("Мія", "unknown"),
    ("Міє?", "unknown"),
    ("asdf", "unknown"),
    ("Пам’ятаєш мене?", "unknown"),
    # == blind test BT: wrong side effects and wrong replies ================================
    # -- feeding: refusals, other people's food, questions and games never feed her --------
    ("Не можу їсти, нема апетиту", "unknown"),
    ("Їсти не хочу", "unknown"),
    ("Не їж стільки солодкого 😄", "unknown"),
    ("Не їж мені мізки 😤", "unknown"),
    ("Навіть поїсти спокійно не дають 😩", "unknown"),
    ("Що ти любиш їсти?", "unknown"),
    ("Мама кличе їсти, зараз прийду", "unknown"),
    ("В Майнкрафті треба їсти, щоб не вмерти", "unknown"),
    ("Мій герой голодний, де взяти їжу?", "unknown"),
    ("Їжа в цій грі виглядає смачно", "unknown"),
    ("Їжу замовив, чекаю кур'єра", "unknown"),
    # ...the same rules, other words.
    ("Я не встиг поїсти", "unknown"),
    ("Не дам тобі печиво", "unknown"),
    ("Я тебе не нагодую", "unknown"),
    ("Ти любиш печиво?", "unknown"),
    ("Мене кличуть обідати", "unknown"),
    ("Як часто треба їсти?", "unknown"),
    ("Я можу їсти борщ хоч щодня", "unknown"),  # answers her idle question
    ("Борщ", "unknown"),
    ("Замовив поїсти, чекаю", "unknown"),
    ("Купила собі печиво", "unknown"),
    ("Мама нагодує кота", "unknown"),
    ("Він поїсть пізніше", "unknown"),
    # ...while a real offer still feeds her.
    ("Тримай яблучко 🍎", "feed"),
    ("Хочеш яблучко?", "feed"),
    ("Ось тобі піца", "feed"),
    ("Яблучко для тебе", "feed"),
    ("Візьми собі печиво", "feed"),
    ("Не хочеш поїсти?", "feed"),
    ("Тобі треба поїсти", "feed"),
    ("Тримай печиво, тільки не їж усе одразу", "feed"),
    ("Хочу тебе нагодувати", "feed"),
    ("*простягає яблучко*", "feed"),
    # -- petting: a refusal or someone else's hug is not petting her -----------------------
    ("Мене ніхто не обіймає 😢", "unknown"),
    ("Не треба мене обіймати 😤", "unknown"),
    ("Не обіймай мене", "unknown"),
    ("Не хочу тебе обіймати", "unknown"),
    ("Мене погладили", "unknown"),
    ("Мама його гладить", "unknown"),
    ("*гладить по голові*", "pet"),  # roleplay: the user's own action
    ("*обіймає міцно* Привіт!", "pet"),
    ("*не обіймає*", "unknown"),
    ("Можна тебе обійняти?", "pet"),
    ("Обіймаю тебе", "pet"),
    ("Не плач, обійму тебе", "pet"),
    # -- sleep: asked to stay up is not "go to bed" -----------------------------------------
    ("Не йди спати, побудь зі мною", "unknown"),
    ("Тобі не час спати, ще рано!", "unknown"),
    ("Не лягай спати", "unknown"),
    ("Я трохи відпочину", "unknown"),  # the user, not a typo for "відпочинь"
    ("Не сиди так довго, йди спати", "rest"),
    # -- quiet: statements about silence or the user's own silence do not mute her ---------
    ("Тихо так на вулиці сьогодні", "unknown"),
    ("Тихо у нас сьогодні", "unknown"),
    ("Тихенько сиджу, граю в майнкрафт", "unknown"),
    ("Він тихенько грає", "unknown"),
    ("Тихіше їдеш — далі будеш", "unknown"),
    ("Обожнюю тишу вночі", "unknown"),
    ("У навушниках тиша, звук пропав", "unknown"),
    ("Тиша стоїть така, що чути годинник", "unknown"),
    ("Я помовчу трохи, треба подумати", "unknown"),
    ("Краще б я помовчав тоді на нараді", "unknown"),
    ("Я замовкну", "unknown"),
    ("Вони замовкли", "unknown"),
    ("Не замовчуй правду", "unknown"),
    ("Не говори мамі, що я грав до третьої ночі 😅", "unknown"),
    ("Не говори нікому", "unknown"),
    ("Не говори татові", "unknown"),
    ("Як перекласти «помовч» англійською?", "unknown"),
    ("Гра «Тиша» класна", "unknown"),
    # ...while asking for quiet still works.
    ("Будь ласка, потихіше", "quiet_on"),
    ("Можеш трохи менше говорити?", "quiet_on"),
    ("Менше говори", "quiet_on"),
    ("Тихіше говори", "quiet_on"),
    ("Тихо, сплю!", "quiet_on"),
    ("Тихенько посидь", "quiet_on"),
    ("Тиша!", "quiet_on"),
    ("Все, тиша", "quiet_on"),
    ("Хвилину тиші, будь ласка", "quiet_on"),
    ("Помовчиш трохи?", "quiet_on"),
    ("Давай помовчимо", "quiet_on"),
    ("Замовкай вже", "quiet_on"),
    ("Не говори зі мною", "quiet_on"),
    ("«Помовч»", "quiet_on"),  # copied from her own hint, quotes and all
    ("Міє, «помовч», будь ласка", "quiet_on"),
    # -- talking again: how she talks is not whether ----------------------------------------
    ("Про це можна говорити годинами", "unknown"),
    ("З тобою можна говорити годинами", "unknown"),
    ("Говори нормально!", "unknown"),
    ("Говори українською", "unknown"),
    ("Не замовкай!", "quiet_off"),
    ("Не затихай", "quiet_off"),
    ("Говори далі", "quiet_off"),
    ("Можеш говорити скільки хочеш", "quiet_off"),
    # -- her name, not anyone's -------------------------------------------------------------
    ("Як звати цього персонажа в грі, не пам'ятаєш?", "unknown"),
    ("Як звати твою маму?", "unknown"),
    ("Як звати твого творця?", "unknown"),
    ("Як мені тебе називати?", "name"),
    ("Як звати тебе?", "name"),
    ("Як тебе звуть, Міє?", "name"),
    # -- the time, not a duration -----------------------------------------------------------
    ("Скільки часу займає ця місія?", "unknown"),
    ("Скільки часу ти вже тут?", "unknown"),
    ("Скільки годин до Нового року?", "unknown"),
    ("Скільки часу пройшло?", "unknown"),
    ("Який час доби в тебе найулюбленіший?", "unknown"),
    ("Котрий час?", "time"),
    ("Скільки щас часу?", "time"),
    ("Скільки часу, не підкажеш?", "time"),
    ("Який зараз час?", "time"),
    # -- a quoted word is mentioned, not said -----------------------------------------------
    ("Як правильно: «дякую» чи «спасибі»?", "unknown"),
    ("Як буде «привіт» японською?", "unknown"),
    ("Привіт! Як буде «кіт» англійською?", "greeting"),
    # -- homographs and idioms --------------------------------------------------------------
    ("Я плачу за інтернет 300 гривень на місяць", "unknown"),
    ("Плачу за квартиру", "unknown"),
    ("Я плачу за тобою", "sad"),
    ("Мені погано видно, що там на екрані", "unknown"),
    ("Мені важко повірити", "unknown"),
    ("Пока не знаю, що робити", "unknown"),
    ("Пока що не знаю", "unknown"),
    ("Пока, Міє", "bye"),
    ("Пока-пока", "bye"),
    ("Мені пора визнати, що я програв", "unknown"),
    ("Мені пора на роботу", "bye"),
    ("Не дякуй 😊", "unknown"),
    ("Це допоможе", "unknown"),
    ("Мама тебе вітає", "unknown"),
    # -- common phrasings that were not understood ------------------------------------------
    ("як твої справи?", "how_are_you"),
    ("Як пройшов твій день?", "how_are_you"),
    ("Як пройшов мій день — жах", "unknown"),
    ("Як спарви?", "how_are_you"),  # typo
    ("Як справу закрити?", "unknown"),
    ("Міє, які в тебе є команди?", "capabilities"),
    ("Йду на роботу, гарного тобі дня!", "bye"),
    ("Спокійної тобі ночі", "bye"),
    ("Ти найкрасивіша", "compliment"),
    ("Наймиліша!", "compliment"),
    ("Ти найтупіша", "unknown"),
    ("дякю", "thanks"),  # typo: a dropped letter in a short word
]

# (her name, message, expected intent): the user renamed her in Settings.
RENAMED: list[tuple[str, str, str]] = [
    ("Кіра", "Кіро, будь тихо", "quiet_on"),
    ("Кіра", "Кіро, тихо", "quiet_on"),
    ("Кіра", "Тихо, Кіро", "quiet_on"),
    ("Кіра", "Як ти, Кіро?", "how_are_you"),
    ("Кіра", "Кіра молодець", "compliment"),
    ("Кіра", "Молодець, Кіро!", "compliment"),
    ("Кіра", "Можеш говорити, Кіро", "quiet_off"),
    ("Кіра", "Кіро, настрій?", "mood"),
    ("Ліна", "Ліно, тихо", "quiet_on"),
    ("Соломія", "Як ти, Соломіє?", "how_are_you"),
    ("Таня", "Таню, тихо", "quiet_on"),
    ("Мар'яна", "Мар’яно, тихо", "quiet_on"),
    ("Ліззі", "Ліззі, тихо", "quiet_on"),
    ("Mia", "Mia, тихо", "quiet_on"),
]


def test_the_corpus_is_big_and_has_no_duplicates():
    messages = [text for text, _ in CORPUS]
    assert len(messages) >= 180
    assert len(set(messages)) == len(messages)


@pytest.mark.parametrize(("text", "intent"), CORPUS)
def test_corpus(text, intent):
    assert detect_intent(text) == intent


@pytest.mark.parametrize(("name", "text", "intent"), RENAMED)
def test_renamed_character(name, text, intent):
    assert detect_intent(text, name) == intent


def test_only_her_current_name_is_an_address():
    # Proves the renamed cases above really depend on the name: "Кіро" means nothing to "Мія".
    assert detect_intent("Кіро, тихо") == "unknown"
    assert detect_intent("Кіро, тихо", "Кіра") == "quiet_on"


@pytest.mark.parametrize(
    ("name", "vocative"),
    [
        ("Мія", "міє"), ("Кіра", "кіро"), ("Соломія", "соломіє"), ("Таня", "таню"), ("Ліна", "ліно"),
        ("Сашко", "сашку"), ("Андрій", "андрію"), ("Ігор", "ігоре"), ("Василь", "василю"),
    ],
)
def test_vocative_forms(name, vocative):
    forms = address_forms(name)
    assert (vocative,) in forms
    assert (name.lower(),) in forms


# -- end to end: side effects through OfflineChatResponder -------------------------------------

OUTCOMES = {"feed": "fed", "pet": "happy", "rest": "sleep"}


class Performer:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, action: str) -> str:
        self.calls.append(action)
        return OUTCOMES[action]


class Chat:
    def __init__(self, bus, db, clock, name="Мія"):
        self.bus = bus
        self.settings = Settings()
        self.settings.character.name = name
        self.performer = Performer()
        self.selector = PhraseSelector(BANK, PhraseHistoryRepository(db), lambda: self.settings, clock,
                                       random.Random(5))
        self.reactions = ReactionEngine(bus, self.selector, lambda: self.settings, clock, random.Random(6))
        self.responder = OfflineChatResponder(self.selector, self.reactions, lambda: self.settings, clock,
                                              perform=self.performer)

    def say(self, text):
        reply = self.responder.respond(text)
        self.bus.dispatch_pending()
        return reply

    def texts(self, key):
        values = {"name": self.settings.character.name}
        out = set()
        for phrase in BANK.phrases(key):
            try:
                out.add(render(phrase.text, values))
            except KeyError:
                continue
        return out


@pytest.fixture
def chat(bus, db, clock):
    clock.set_now(datetime(2026, 3, 10, 14, 7).timestamp())
    return Chat(bus, db, clock)


@pytest.mark.parametrize(
    ("text", "performed", "reply_action"),
    [
        ("Я втомився, йди спати", ["rest"], "rest"),
        ("Я теж втомився, йди спати", ["rest"], "rest"),
        ("Йди спати, мені теж пора", ["rest"], "rest"),
        ("Нам пора спати", ["rest"], "rest"),
        ("Міє, йди спати", ["rest"], "rest"),
        ("Я приніс поїсти", ["feed"], "fed"),
        ("Я приготувала поїсти", ["feed"], "fed"),
        ("Тримай печиво, Міє", ["feed"], "fed"),
        ("Погладжу тебе, Міє", ["pet"], "petted"),
        # The user's own hunger, bedtime or sadness never acts on her.
        ("Я голодний", [], ""),
        ("Я хочу їсти", [], ""),
        ("Я голодна, піду поїсти", [], ""),
        ("Мені треба поїсти", [], ""),
        ("Я з’їв печиво", [], ""),
        ("Мені пора спати", [], ""),
        ("Мені погано, обійми мене", [], ""),
        ("Мені сумно, розкажи жарт", [], ""),
        # BT: refusals, other people's food, games and questions never act on her.
        ("Не можу їсти, нема апетиту", [], ""),
        ("Їсти не хочу", [], ""),
        ("Не їж стільки солодкого 😄", [], ""),
        ("Не їж мені мізки 😤", [], ""),
        ("Що ти любиш їсти?", [], ""),
        ("Мама кличе їсти, зараз прийду", [], ""),
        ("В Майнкрафті треба їсти, щоб не вмерти", [], ""),
        ("Мій герой голодний, де взяти їжу?", [], ""),
        ("Їжа в цій грі виглядає смачно", [], ""),
        ("Їжу замовив, чекаю кур'єра", [], ""),
        ("Навіть поїсти спокійно не дають 😩", [], ""),
        ("Мене ніхто не обіймає 😢", [], ""),
        ("Не треба мене обіймати 😤", [], ""),
        ("Не обіймай мене", [], ""),
        ("Не йди спати, побудь зі мною", [], ""),
        ("Тобі не час спати, ще рано!", [], ""),
        ("Я трохи відпочину", [], ""),
        # ...and real offers or roleplay still do.
        ("Тримай яблучко 🍎", ["feed"], "fed"),
        ("*гладить по голові*", ["pet"], "petted"),
    ],
)
def test_actions_happen_only_when_asked(chat, text, performed, reply_action):
    reply = chat.say(text)
    assert chat.performer.calls == performed
    assert reply.action == reply_action


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Мені сумно, розкажи жарт", "idle.joke"),
        ("Мені сумно", "chat.user_sad"),
        ("Мені погано, обійми мене", "chat.user_sad"),
        ("Мій персонаж помер", "chat.unknown"),
        ("Як ти, Міє?", "chat.how_are_you"),
        ("Молодець, Міє!", "chat.compliment"),
        ("Покажи команди", "chat.capabilities"),
        ("Мені пора спати", "chat.bye"),
        # BT: honest "I don't understand" instead of a wrong answer...
        ("Як звати твою маму?", "chat.unknown"),
        ("Скільки часу займає ця місія?", "chat.unknown"),
        ("Як буде «привіт» японською?", "chat.unknown"),
        ("Як правильно: «дякую» чи «спасибі»?", "chat.unknown"),
        ("Я плачу за інтернет 300 гривень на місяць", "chat.unknown"),
        ("Мені погано видно, що там на екрані", "chat.unknown"),
        ("Пока не знаю, що робити", "chat.unknown"),
        ("Мені пора визнати, що я програв", "chat.unknown"),
        ("Не дякуй 😊", "chat.unknown"),
        # ...and the right answer where she used to give up.
        ("як твої справи?", "chat.how_are_you"),
        ("Як мені тебе називати?", "chat.name"),
        ("Міє, які в тебе є команди?", "chat.capabilities"),
        ("Йду на роботу, гарного тобі дня!", "chat.bye"),
        ("Ти найкрасивіша", "chat.compliment"),
        ("дякю", "chat.thanks"),
    ],
)
def test_replies_come_from_the_right_pool(chat, text, key):
    assert chat.say(text).text in chat.texts(key)


@pytest.mark.parametrize(
    "text",
    ["Міє, будь тихо", "Будь тихо", "Міє, тихо", "Тепер тихо", "Можна тихо?", "тсс", "Відстань",
     "Будь ласка, потихіше", "Можеш трохи менше говорити?", "Тиша!", "«Помовч»"],
)
def test_asking_for_quiet_turns_it_on(chat, text):
    reply = chat.say(text)
    assert reply.action == "quiet_on"
    assert chat.reactions.quiet_active


@pytest.mark.parametrize(
    "text",
    ["Чому так тихо?", "Тихо тут якось", "Хм, дай подумати", "Не відволікайся", "Мій персонаж помер",
     # BT
     "Тихо так на вулиці сьогодні", "Тихенько сиджу, граю в майнкрафт", "Обожнюю тишу вночі",
     "У навушниках тиша, звук пропав", "Я помовчу трохи, треба подумати", "Краще б я помовчав тоді на нараді",
     "Не говори мамі, що я грав до третьої ночі 😅", "Як перекласти «помовч» англійською?",
     "Тихіше їдеш — далі будеш"],
)
def test_quiet_mode_is_not_turned_on_by_accident(chat, text):
    reply = chat.say(text)
    assert reply.action == ""
    assert not chat.reactions.quiet_active


@pytest.mark.parametrize(
    "text",
    ["Можеш говорити, Міє", "Міє, можеш говорити", "Можеш говорити далі", "Можеш говорити, дякую",
     "Мія, можна говорити", "Не мовчи!", "Скасуй режим тиші", "Не замовкай!"],
)
def test_asking_her_to_talk_turns_quiet_off(chat, text):
    chat.reactions.set_quiet(60)
    reply = chat.say(text)
    assert reply.action == "quiet_off"
    assert not chat.reactions.quiet_active


@pytest.mark.parametrize(
    "text",
    ["Ти можеш говорити українською?", "Чому так тихо?", "Вже можна їсти?", "Мені сумно",
     "Про це можна говорити годинами", "Говори нормально!"],
)
def test_quiet_mode_survives_other_messages(chat, text):
    chat.reactions.set_quiet(60)
    reply = chat.say(text)
    assert reply.action == ""
    assert chat.reactions.quiet_active


def test_a_renamed_character_answers_to_her_new_name(bus, db, clock):
    chat = Chat(bus, db, clock, name="Кіра")
    reply = chat.say("Кіро, тихо")
    assert reply.action == "quiet_on"
    assert chat.reactions.quiet_active
    reply = chat.say("Можеш говорити, Кіро")
    assert reply.action == "quiet_off"
    assert not chat.reactions.quiet_active
    assert chat.say("Кіро, йди спати").action == "rest"
    assert chat.performer.calls == ["rest"]
