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
        ("на, поїж", "feed", "full", "feed", "fed"),
        ("нагодую тебе", "feed", "starving_fed", "feed", "fed"),
        ("погладжу", "pet", "happy", "pet", "petted"),
        ("обійми", "pet", "too_much", "pet", "petted"),
        ("йди спати", "rest", "sleep", "rest", "rest"),
        ("відпочинь", "rest", "not_tired", "rest", "rest"),
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
    assert reply.action == "fed"


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
