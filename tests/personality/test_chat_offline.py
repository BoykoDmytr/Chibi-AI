import logging
import random
from datetime import datetime

import pytest

from chibi.core.config import Settings
from chibi.core.events import QuietModeChanged
from chibi.memory.repositories import PhraseHistoryRepository
from chibi.personality.chat_offline import FALLBACKS, OfflineChatResponder, detect_intent
from chibi.personality.phrases import PhraseBank, PhraseSelector
from chibi.personality.reactions import ReactionEngine
from chibi.personality.text_uk import render
from chibi.settings.facade import ChatReply
from chibi.tamagotchi.needs import Needs

BANK = PhraseBank.load()


class Performer:
    def __init__(self, outcome="fed"):
        self.outcome = outcome
        self.calls: list[str] = []

    def __call__(self, action: str) -> str:
        self.calls.append(action)
        return self.outcome


class Chat:
    def __init__(self, bus, db, clock, bank=BANK, perform=None, needs=None):
        self.bus, self.clock = bus, clock
        self.settings = Settings()
        self.selector = PhraseSelector(bank, PhraseHistoryRepository(db), lambda: self.settings, clock,
                                       random.Random(5))
        self.reactions = ReactionEngine(bus, self.selector, lambda: self.settings, clock, random.Random(6))
        self.responder = OfflineChatResponder(self.selector, self.reactions, lambda: self.settings, clock,
                                              perform=perform, needs=needs)

    def texts(self, key, **variables):
        values = {"name": self.settings.character.name, **variables}
        out = set()
        for phrase in BANK.phrases(key):
            try:
                out.add(render(phrase.text, values))
            except KeyError:
                continue
        return out

    def say(self, text):
        reply = self.responder.respond(text)
        self.bus.dispatch_pending()
        return reply


@pytest.fixture
def chat(bus, db, clock):
    clock.set_now(datetime(2026, 3, 10, 14, 7).timestamp())
    return Chat(bus, db, clock)


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Привіт!", "greeting"), ("привііііт", "greeting"), ("пирвіт", "greeting"), ("Хай!", "greeting"),
        ("хей, ти тут?", "greeting"), ("Добрий день", "greeting"), ("доброго ранку", "greeting"),
        ("Добрий вечір!", "greeting"), ("вітаю", "greeting"), ("hello", "greeting"), ("привiт", "greeting"),
        ("Як справи?", "how_are_you"), ("як ти?", "how_are_you"), ("привіт, як справи", "how_are_you"),
        ("Як тебе звати?", "name"), ("хто ти?", "name"), ("як твоє ім’я", "name"),
        ("Що ти вмієш?", "capabilities"), ("допомога", "capabilities"), ("help", "capabilities"),
        ("розкажи жарт", "joke"), ("анекдот!", "joke"), ("розсміши мене", "joke"),
        ("розкажи щось", "thought"), ("спитай мене щось", "question"),
        ("помовч", "quiet_on"), ("Тихо!", "quiet_on"), ("тихіше, будь ласка", "quiet_on"),
        ("не заважай", "quiet_on"), ("замовкни", "quiet_on"), ("тиша", "quiet_on"),
        ("дай спокій", "quiet_on"),
        ("не говори", "quiet_on"), ("можеш не говорити", "quiet_on"), ("помовчи, дурна", "quiet_on"),
        ("можеш говорити", "quiet_off"), ("говори", "quiet_off"), ("повертайся", "quiet_off"),
        ("вже можна", "quiet_off"),
        ("дякую!", "thanks"), ("дякс", "thanks"), ("спасибі", "thanks"), ("дкяую", "thanks"),
        ("ти класна", "compliment"), ("ти найкраща", "compliment"), ("люблю тебе", "compliment"),
        ("ти не тупа, ти розумниця", "compliment"),
        ("ти тупа", "rude"), ("дурепа", "rude"), ("ненавиджу тебе", "rude"),
        ("на, поїж", "feed"), ("тримай смаколик", "feed"), ("нагодую тебе", "feed"), ("хочеш їсти?", "feed"),
        ("погладжу тебе", "pet"), ("обійми мене", "pet"), ("гладь", "pet"),
        ("йди спати", "rest"), ("відпочинь", "rest"), ("лягай спати", "rest"),
        ("котра година?", "time"), ("скільки зараз часу", "time"),
        ("як ти себе почуваєш?", "mood"), ("який у тебе настрій", "mood"),
        ("бувай", "bye"), ("па-па", "bye"), ("до зустрічі!", "bye"), ("на добраніч", "bye"), ("поки", "bye"),
        ("asdf", "unknown"), ("", "unknown"), ("   ", "unknown"), ("?!", "unknown"),
        ("поки що не знаю", "unknown"), ("хай буде так", "unknown"), ("здоров'я", "unknown"),
        ("розкажи про квантову фізику", "unknown"),
        ("я там бував", "unknown"), ("я повертаюся", "unknown"),  # one edit away, but real words
    ],
)
def test_detect_intent(text, intent):
    assert detect_intent(text) == intent


def test_greeting_reply(chat):
    reply = chat.say("Привіт!")
    assert isinstance(reply, ChatReply)
    assert reply.text in chat.texts("chat.greeting")
    assert reply.action == ""


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("як справи?", "chat.how_are_you"),
        ("що ти вмієш", "chat.capabilities"),
        ("розкажи жарт", "idle.joke"),
        ("дякую", "chat.thanks"),
        ("ти чудова", "chat.compliment"),
        ("ти тупа", "chat.rude"),
        ("бувай", "chat.bye"),
        ("qwerty uiop", "chat.unknown"),
    ],
)
def test_replies_come_from_the_intent_key(chat, text, key):
    reply = chat.say(text)
    assert reply.text in chat.texts(key)
    assert reply.action == ""


def test_name_uses_her_current_name(chat):
    chat.settings.character.name = "Ліна"
    reply = chat.say("як тебе звати?")
    assert "Ліна" in reply.text
    assert reply.text in chat.texts("chat.name")


def test_time_reply(chat):
    reply = chat.say("котра година?")
    assert "14:07" in reply.text
    assert reply.text in chat.texts("chat.time", time="14:07")


def test_quiet_on_and_off(chat, bus, clock):
    events = []
    bus.subscribe(QuietModeChanged, events.append)
    chat.settings.behavior.quiet_minutes = 45
    reply = chat.say("помовч, будь ласка")
    assert reply.action == "quiet_on"
    assert reply.text in chat.texts("quiet.ack")
    assert chat.reactions.quiet_active
    assert chat.reactions.quiet_until == pytest.approx(clock.now() + 45 * 60)
    assert events[-1].enabled is True
    reply = chat.say("можеш говорити")
    assert reply.action == "quiet_off"
    assert reply.text in chat.texts("quiet.end")
    assert not chat.reactions.quiet_active
    assert events[-1].enabled is False


@pytest.mark.parametrize(
    ("text", "action", "outcome", "performed", "reply_action"),
    [
        ("тримай смаколик", "feed", "fed", "feed", "fed"),
        ("нагодую тебе", "feed", "starving_fed", "feed", "fed"),
        ("погладжу", "pet", "happy", "pet", "petted"),
        ("йди спати", "rest", "sleep", "rest", "rest"),
        # Refusals: the chat must not print "she ate / purrs / went to rest".
        ("на, поїж", "feed", "full", "feed", ""),
        ("обійми", "pet", "too_much", "pet", ""),
        ("відпочинь", "rest", "not_tired", "rest", ""),
    ],
)
def test_actions_perform_and_reply_with_the_outcome(bus, db, clock, text, action, outcome, performed,
                                                    reply_action):
    performer = Performer(outcome)
    chat = Chat(bus, db, clock, perform=performer)
    reply = chat.say(text)
    assert performer.calls == [performed]
    assert reply.action == reply_action
    assert reply.text in chat.texts(f"interaction.{action}.{outcome}")


def test_rest_reply_carries_the_sleep_state(bus, db, clock):
    chat = Chat(bus, db, clock, perform=Performer("sleep"))
    reply = chat.say("йди спати")
    phrase_states = {p.state for p in BANK.phrases("interaction.rest.sleep")}
    assert reply.state in phrase_states


def test_action_without_perform_is_honest(chat):
    reply = chat.say("тримай смаколик")
    assert reply.text == FALLBACKS["action_unavailable"]
    assert reply.action == ""


def test_action_failure_is_contained(bus, db, clock, caplog):
    def broken(_action):
        raise RuntimeError("tamagotchi asleep")

    chat = Chat(bus, db, clock, perform=broken)
    reply = chat.say("погладжу")
    assert reply.text == FALLBACKS["action_unavailable"]
    assert "chat action pet failed" in caplog.text


def test_unexpected_outcome_still_replies(bus, db, clock):
    chat = Chat(bus, db, clock, perform=Performer("weird"))
    reply = chat.say("тримай смаколик")
    assert reply.text == FALLBACKS["feed"]
    assert reply.action == ""  # unknown outcome: do not claim she ate


@pytest.mark.parametrize(
    ("text", "outcome", "fallback", "action"),
    [
        ("тримай смаколик", "full", "feed.full", ""),
        ("погладжу", "too_much", "pet.too_much", ""),
        ("йди спати", "not_tired", "rest.not_tired", ""),
        ("йди спати", "sleep", "rest", "rest"),
    ],
)
def test_refusals_without_a_phrase_bank_do_not_claim_success(bus, db, clock, text, outcome, fallback, action):
    chat = Chat(bus, db, clock, bank=PhraseBank(), perform=Performer(outcome))
    reply = chat.say(text)
    assert (reply.text, reply.action) == (FALLBACKS[fallback], action)


@pytest.mark.parametrize(
    ("snapshot", "key"),
    [
        (Needs(energy=10).to_dict(), "chat.mood.sleepy"),
        (Needs(hunger=90).to_dict(), "chat.mood.hungry"),
        (Needs(boredom=90).to_dict(), "chat.mood.bored"),
        (Needs(happiness=10).to_dict(), "chat.mood.sad"),
        (Needs(happiness=95).to_dict(), "chat.mood.happy"),
        (Needs(happiness=70).to_dict(), "chat.mood.cheerful"),
        (Needs(happiness=50).to_dict(), "chat.mood.calm"),
        ({"mood": "Весела", "hunger": 99.0}, "chat.mood.cheerful"),  # an explicit mood wins
    ],
)
def test_mood_reply_follows_needs(bus, db, clock, snapshot, key):
    chat = Chat(bus, db, clock, needs=lambda: snapshot)
    assert chat.say("як ти себе почуваєш?").text in chat.texts(key)


def test_mood_from_a_view_object(bus, db, clock):
    class View:
        mood = "Сонна"

    chat = Chat(bus, db, clock, needs=View)
    assert chat.say("який настрій?").text in chat.texts("chat.mood.sleepy")


def test_mood_without_needs_falls_back_to_how_are_you(chat):
    assert chat.say("як ти себе почуваєш?").text in chat.texts("chat.how_are_you")


def test_mood_needs_failure_falls_back(bus, db, clock):
    def broken():
        raise RuntimeError("no needs")

    chat = Chat(bus, db, clock, needs=broken)
    assert chat.say("як настрій?").text in chat.texts("chat.how_are_you")


def test_empty_bank_uses_builtin_fallbacks(bus, db, clock):
    clock.set_now(datetime(2026, 3, 10, 9, 30).timestamp())
    chat = Chat(bus, db, clock, bank=PhraseBank())
    chat.settings.character.name = "Ліна"
    assert chat.say("як тебе звати").text == "Мене звати Ліна."
    assert chat.say("котра година").text == "Зараз 09:30."
    assert chat.say("бла-бла").text == FALLBACKS["unknown"]
    reply = chat.say("помовч")
    assert (reply.text, reply.action) == (FALLBACKS["quiet_on"], "quiet_on")
    assert chat.reactions.quiet_active


def test_replies_are_solicited_and_recorded(chat, db):
    chat.say("розкажи жарт")
    ((_phrase_id, key, _at),) = PhraseHistoryRepository(db).recent()
    assert key == "idle.joke"


def test_jokes_do_not_repeat_back_to_back(chat):
    jokes = [chat.say("жарт").text for _ in range(10)]
    assert len(set(jokes)) == 10


def test_user_text_is_never_logged(bus, db, clock, caplog):
    def broken_needs():
        raise RuntimeError("boom")

    def broken_perform(_action):
        raise RuntimeError("boom")

    chat = Chat(bus, db, clock, needs=broken_needs, perform=broken_perform)
    secret = "мій пароль qwerty123"
    with caplog.at_level(logging.DEBUG):
        for text in (secret, f"привіт {secret}", f"погладжу {secret}", f"як настрій {secret}"):
            chat.say(text)
    assert caplog.records  # something was logged ...
    assert all("qwerty123" not in record.getMessage() for record in caplog.records)  # ... but not the text
    assert "qwerty123" not in caplog.text


def test_non_text_input_is_unknown(chat):
    assert chat.say(None).text in chat.texts("chat.unknown")


def test_respond_never_raises(bus, db, clock, monkeypatch):
    chat = Chat(bus, db, clock)

    def explode(*_args, **_kwargs):
        raise RuntimeError("selector broke")

    monkeypatch.setattr(chat.selector, "say", explode)
    reply = chat.say("привіт")
    assert reply.text == FALLBACKS["unknown"]


# -- review regressions: who a message is about ------------------------------------------------


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        # UK-1: asking her to talk, or cancelling quiet mode, never turns it on.
        ("Не мовчи!", "quiet_off"), ("Не треба тиші", "quiet_off"), ("Вимкнути тишу", "quiet_off"),
        ("Скасуй режим тиші", "quiet_off"), ("Досить тиші", "quiet_off"),
        ("Мені тиша не потрібна", "quiet_off"),
        ("Тиша набридла", "quiet_off"), ("Годі мовчати", "quiet_off"), ("Не треба мовчати", "quiet_off"),
        ("Чому так тихо?", "unknown"), ("Тихо тут сьогодні", "unknown"),
        ("Не вимикай режим тиші", "quiet_on"), ("Говори тихо", "quiet_on"),
        ("Будь ласка, тихіше", "quiet_on"), ("Вимкни, будь ласка, режим тиші", "quiet_off"),
        ("Тиша набридла, поговори зі мною", "quiet_off"), ("Тиша мені більше не потрібна", "quiet_off"),
        # ...and a request for quiet that mentions "тиша" never lifts it.
        ("Досить говорити, тиша!", "quiet_on"), ("Тиша, не треба нічого казати", "quiet_on"),
        ("Не треба говорити, потрібна тиша", "quiet_on"), ("Вимкни музику, мені потрібна тиша", "quiet_on"),
        ("Досить балакати", "quiet_on"), ("Перестань говорити", "quiet_on"),
        # UK-2: more ways to ask for quiet; ambiguous ones stay out.
        ("Тсс", "quiet_on"), ("Тссс!", "quiet_on"), ("Ш-ш-ш", "quiet_on"), ("ш-ш", "quiet_on"),
        ("Шшш", "quiet_on"), ("Ша!", "quiet_on"), ("Тсс, я на дзвінку", "quiet_on"), ("Відстань", "quiet_on"),
        ("Відстань від мене", "quiet_on"), ("Не відволікай мене, я працюю", "quiet_on"),
        ("Дай поспати", "quiet_on"), ("Дай попрацювати", "quiet_on"),
        ("Не зараз", "unknown"), ("Іди геть", "unknown"), ("Зникни", "unknown"),
        ("Яка відстань до Місяця?", "unknown"), ("Відстань 5 км", "unknown"), ("Не говори так!", "unknown"),
        # UK-7: questions do not lift quiet mode.
        ("Ти можеш говорити українською?", "unknown"), ("Ти можеш розмовляти англійською?", "unknown"),
        ("Вже можна їсти?", "unknown"), ("Можеш знову говорити", "quiet_off"),
        ("Тепер можна говорити", "quiet_off"),
        # UK-5: the user's mood is not a question about hers; compliments need an addressee.
        ("У мене поганий настрій", "sad"), ("Як твій настрій?", "mood"), ("Як у тебе настрій?", "mood"),
        ("Все супер", "good"), ("У мене все класно", "good"), ("Все добре, дякую", "good"),
        ("Супер!", "good"),
        ("Чудово, а в тебе?", "how_are_you"), ("Добре, а ти як?", "how_are_you"),
        ("Гарна погода сьогодні", "unknown"), ("Ця гра чудова", "unknown"), ("Добре", "unknown"),
        ("Ти така мила", "compliment"), ("Яка ти гарна", "compliment"), ("Розумничка!", "compliment"),
        ("Як мило!", "compliment"),
        # UK-6: self-criticism and complaints about a game are not insults to her.
        ("Я такий дурень, знову забув зберегтися", "sad"), ("Цей бос тупий", "unknown"),
        ("Ця гра відстій", "unknown"), ("Ідіот!", "unknown"), ("Ти дурна", "rude"), ("Ти відстій", "rude"),
        ("Я тебе ненавиджу", "rude"), ("Ти мене дістала", "rude"), ("ти ж не дурна", "unknown"),
        # UK-8: other misreads.
        ("Я хочу їсти", "unknown"), ("Я голодна, піду поїсти", "unknown"), ("Мені треба поїсти", "unknown"),
        ("Я тебе нагодую", "feed"), ("Поїж, а я теж поїм", "feed"), ("Я тебе люблю", "compliment"),
        ("Спокійної ночі", "bye"), ("Доброї ночі", "bye"), ("Гарного дня!", "bye"), ("Гарного вечора", "bye"),
        ("Доброго дня!", "greeting"), ("Доброго вечора", "greeting"),
        ("Моя команда знову злила катку", "unknown"), ("Які в тебе команди?", "capabilities"),
        ("Як ти думаєш, піца з ананасами норм?", "unknown"), ("Як ти там?", "how_are_you"),
        ("Дякую за допомогу", "thanks"),
        ("Мені пора спати", "bye"), ("Йди спати, а я ще пограю", "rest"),
        # UK-14: sad news gets support, not "sounds interesting".
        ("Мені сумно", "sad"), ("Мені погано", "sad"), ("У мене помер кіт", "sad"), ("Мій кіт помер", "sad"),
        ("Мені сумно, обійми мене", "sad"), ("Не дуже, а ти?", "sad"), ("Я помер на босі", "unknown"),
        ("Мені не сумно", "unknown"),
        # Typos are corrected, but the context rules still apply.
        ("ти чудвоа", "compliment"), ("ця гра чудвоа", "unknown"), ("нагдую тебе", "feed"),
    ],
)
def test_detect_intent_reads_who_it_is_about(text, intent):
    assert detect_intent(text) == intent


def test_asking_her_to_talk_lifts_quiet_mode(chat, bus):
    events = []
    bus.subscribe(QuietModeChanged, events.append)
    for text in ("Не мовчи!", "Скасуй режим тиші", "Досить тиші"):
        chat.reactions.set_quiet(60)
        reply = chat.say(text)
        assert reply.action == "quiet_off", text
        assert not chat.reactions.quiet_active, text
        assert events[-1].enabled is False


def test_a_question_does_not_lift_quiet_mode(chat):
    chat.reactions.set_quiet(60)
    for text in ("Ти можеш говорити українською?", "Вже можна їсти?"):
        reply = chat.say(text)
        assert reply.action == "", text
        assert chat.reactions.quiet_active, text


def test_the_users_hunger_does_not_feed_her(bus, db, clock):
    performer = Performer("fed")
    chat = Chat(bus, db, clock, perform=performer)
    chat.say("Я хочу їсти")
    chat.say("Я голодна, піду поїсти")
    assert performer.calls == []


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Мені сумно", "chat.user_sad"),
        ("У мене поганий настрій", "chat.user_sad"),
        ("У мене все класно", "chat.user_good"),
    ],
)
def test_the_users_feelings_get_their_own_replies(bus, db, clock, text, key):
    # Even with needs wired in: "у мене поганий настрій" is not a question about hers.
    chat = Chat(bus, db, clock, needs=lambda: Needs(happiness=95).to_dict())
    reply = chat.say(text)
    assert reply.text in chat.texts(key)
    assert reply.action == ""


def test_feeling_fallbacks_without_a_phrase_bank(bus, db, clock):
    chat = Chat(bus, db, clock, bank=PhraseBank())
    assert chat.say("Мені сумно").text == FALLBACKS["sad"]
    assert chat.say("Все супер").text == FALLBACKS["good"]


def test_rest_reply_state_is_always_sleep(bus, db, clock):
    # integ-3: any other state would wake her right after "йди спати".
    chat = Chat(bus, db, clock, perform=Performer("sleep"))
    states = {chat.say("йди спати").state for _ in range(len(BANK.phrases("interaction.rest.sleep")) * 2)}
    assert states == {"sleep"}
