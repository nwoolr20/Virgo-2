from virgo2.salience import clamp_salience, decay_salience, estimate_salience, reinforce_salience


def test_salience_behaviors():
    assert estimate_salience("remember this") > estimate_salience("hello")
    assert estimate_salience("hello", role="assistant") < estimate_salience("hello", role="user")
    assert decay_salience(2.0, rate=0.5) < 2.0
    assert reinforce_salience(1.0) > 1.0
    assert clamp_salience(100.0) <= 5.0
