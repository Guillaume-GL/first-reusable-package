from reusablepackage.lib import try_me


def test_try_me():
    assert try_me() == '❌ First try, second guess ❔ ✅'
