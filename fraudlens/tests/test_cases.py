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
