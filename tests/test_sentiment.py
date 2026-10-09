from lcw_analytics.sentiment import rule_based_sentiment, _temizle


def test_temizle():
    assert _temizle("Merhaba, Dünya!") == "merhaba  dünya "
    assert _temizle(None) == ""


def test_positive():
    label, score = rule_based_sentiment("Ürün çok güzel ve kaliteli")
    assert label == "positive"
    assert score > 0


def test_negative():
    label, score = rule_based_sentiment("Çok kötü, berbat bir ürün")
    assert label == "negative"
    assert score < 0


def test_negation():
    label, _ = rule_based_sentiment("Bu ürün güzel değil")
    assert label == "negative"


def test_neutral():
    label, _ = rule_based_sentiment("Ürün bugün geldi")
    assert label == "neutral"