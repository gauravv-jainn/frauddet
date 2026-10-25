import pytest

from fraudkit.cases import CaseStore


def test_case_lifecycle_and_audit(tmp_path):
    s = CaseStore(tmp_path / "c.sqlite")
    cid = s.add_case(
        trans_num="t1",
        cc_num=1,
        txn_time="2020-07-01 03:00",
        amount=900.0,
        category="shopping_net",
        merchant="m",
        score=0.93,
        band="Critical",
        reasons=["Amount is 9.0x usual"],
    )
    assert (
        s.add_case(
            trans_num="t1",
            cc_num=1,
            txn_time="x",
            amount=1,
            category="c",
            merchant="m",
            score=0.5,
            band="High",
            reasons=[],
        )
        is None
    )  # no duplicates
    s.add_case(
        trans_num="t2",
        cc_num=2,
        txn_time="2020-07-01 04:00",
        amount=20.0,
        category="grocery_pos",
        merchant="m2",
        score=0.4,
        band="High",
        reasons=[],
    )
    s.update_status(cid, "Escalate", "check with bank")
    s.update_status(cid, "Confirm fraud", "customer confirmed")
    assert s.cases().set_index("trans_num").loc["t2", "assignee"] is None  # NULL must not become NaN (truthy)
    cases = s.cases()
    assert cases.loc[cases.case_id == cid, "status"].item() == "confirmed_fraud"
    assert cases.loc[cases.case_id == cid, "reasons"].item() == ["Amount is 9.0x usual"]
    trail = s.audit(cid)
    assert list(trail["action"])[::-1] == ["created", "Escalate", "Confirm fraud"]
    st = s.stats()
    assert st["confirmed_fraud"] == 1 and st["open"] == 1 and st["agreement_rate"] == 1.0
    with pytest.raises(ValueError):
        s.update_status(cid, "Delete")


def _case(s, tn, cc=4000123412341234, score=0.9):
    return s.add_case(
        trans_num=tn,
        cc_num=cc,
        txn_time="2020-07-01",
        amount=100.0,
        category="c",
        merchant="m",
        score=score,
        band="High",
        reasons=[],
        expected_loss=90.0,
        decision="review",
    )


def test_comments_bulk_actions_and_sla(tmp_path):
    s = CaseStore(tmp_path / "c.sqlite")
    ids = [_case(s, f"t{i}") for i in range(3)]
    s.add_comment(ids[0], "kim", "Called the customer, no answer")
    assert s.comments(ids[0])["text"].tolist() == ["Called the customer, no answer"]
    with pytest.raises(ValueError):
        s.add_comment(ids[0], "kim", "   ")
    s.bulk_assign(ids, "kim", actor="kim")
    assert s.stats(sla_hours=24, user="kim")["my_open"] == 3
    assert s.stats(sla_hours=0.000001, user="kim")["my_overdue"] == 3  # everything is past a tiny SLA
    s.bulk_update(ids[:2], "Mark genuine", "batch", actor="kim")
    st = s.stats(sla_hours=24, user="kim")
    assert st["genuine"] == 2 and st["my_open"] == 1
    df = s.cases()
    assert df["card_ref"].notna().all() and set(df["decision"]) == {"review"}


def test_watchlist_add_remove_and_sets(tmp_path):
    from fraudkit.db import Database
    from fraudkit.security import card_hash
    from fraudkit.watchlist import WatchlistStore

    w = WatchlistStore(Database(tmp_path / "w.sqlite"))
    assert w.add_card(4000123412341234, "confirmed fraud", actor="kim")
    assert not w.add_card(4000123412341234, "again", actor="kim")  # already active
    assert w.add_merchant("Bad Shop", "chargebacks", actor="kim")
    sets = w.active_sets()
    assert sets["cards"] == {card_hash(4000123412341234)} and sets["merchants"] == {"Bad Shop"}
    entries = w.entries()
    assert "4000123412341234" not in entries.to_string()  # only masked label and hash are stored
    w.remove(int(entries.loc[entries["kind"] == "merchant", "id"].iloc[0]), actor="kim")
    assert w.active_sets()["merchants"] == set()
    assert w.is_card_blocked(4000123412341234)
