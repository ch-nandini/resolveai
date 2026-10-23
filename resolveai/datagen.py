"""Synthetic world + labeled eval cases.

Each case comes from a scenario with a ground-truth label decided *by the scenario author from the
written policy*, independent of the agent's code: approve (with exact amount), deny, or escalate.
That's what makes "escalation accuracy" a real number instead of a tautology.
"""
import random
from datetime import timedelta

from . import config

NOW = config.SIM_NOW

ITEMS = ["Wireless Earbuds", "Ceramic Mug Set", "Running Shoes", "Desk Lamp", "Yoga Mat", "Bluetooth Speaker",
         "Winter Jacket", "Coffee Grinder", "Phone Case", "Backpack", "Smart Watch", "Air Fryer",
         "27in Monitor", "Office Chair", "Cast Iron Skillet", "Hiking Boots", "Throw Blanket", "Mechanical Keyboard"]
FIRST = ["Ava", "Liam", "Priya", "Noah", "Mei", "Diego", "Sara", "Omar", "Chloe", "Ravi", "Emma", "Kenji",
         "Fatima", "Lucas", "Zoe", "Arjun", "Nina", "Mateo", "Hana", "Ethan"]
LAST = ["Patel", "Kim", "Garcia", "Nguyen", "Smith", "Chen", "Okafor", "Silva", "Rossi", "Khan", "Lee", "Brown"]

MSG = {
    "not_delivered": [
        "My {item} never arrived. Can I get a refund?",
        "Tracking says it's on the way but it's been weeks and my {item} never showed up.",
        "I never received my order. Where is my package?",
        "The {item} hasn't arrived and I'm still waiting. Please refund me.",
        "Tracking shows delivered but there was nothing at my door. I never got the {item}.",
        "Package is lost I think, the {item} never came.",
    ],
    "damaged": [
        "The {item} arrived broken. I'd like a refund.",
        "Opened the box and the {item} was cracked.",
        "My {item} stopped working after two days, it's defective.",
        "The {item} came damaged, the box was crushed.",
        "Received the {item} today and it's shattered.",
        "The {item} is faulty, it doesn't work at all.",
    ],
    "wrong_item": [
        "You sent the wrong {item}, I ordered a different color.",
        "I got a completely different item instead of the {item} I ordered.",
        "This isn't what I ordered. Wrong size on the {item}.",
        "Received the wrong product, not the {item}.",
        "My order was mixed up, this is not what I ordered.",
    ],
    "late_delivery": [
        "My {item} arrived a week late. I needed it for a trip.",
        "The order was delayed by several days past the delivery date.",
        "Delivery took forever, the {item} came way past the promised date.",
        "The {item} showed up late, days after you said it would.",
    ],
    "changed_mind": [
        "I changed my mind about the {item}, can I return it for a refund?",
        "I don't need the {item} anymore. Refund please.",
        "Ordered the {item} by mistake, I'd like to return it.",
        "Found the {item} cheaper elsewhere, want to return it.",
        "I don't like the {item}, I want to return it and get my money back.",
    ],
    "ambiguous": [
        "Hi, I have an issue with my order, please help.",
        "Not happy with this purchase at all.",
        "Can someone look into my order? I want my money back.",
        "The {item} is not what I was hoping for.",
        "Something is off with the {item}. What can you do?",
    ],
}


# Indirect / messy phrasings: no obvious keywords. These are where a real LLM should beat the
# keyword baseline. Mixed in at INDIRECT_RATE so the eval measures language understanding.
INDIRECT = {
    "not_delivered": [
        "Porch was empty when I got home and my neighbors don't have it either. Order for the {item}.",
        "Its been 3 weeks, no {item}, no box, nothing. what is going on",
        "The driver must have dropped it somewhere else because I definitely don't have the {item}.",
        "Checked the mailroom twice, they have no record of my {item}.",
    ],
    "damaged": [
        "The {item} has a spiderweb across the front, looks like it was dropped.",
        "Took the {item} out of the box and a piece snapped right off.",
        "the {item} won't turn on no matter what i do, charged it overnight",
        "There's a huge dent and scratches all over the {item}, not usable.",
    ],
    "wrong_item": [
        "Ordered black, got navy. Need this sorted for the {item}.",
        "The label says {item} but what's inside is a totally different product.",
        "I asked for a medium and a small showed up.",
        "This is someone else's order I think, it's not my {item}.",
    ],
    "late_delivery": [
        "The {item} finally showed up 6 days after the date on my confirmation email.",
        "Paid for fast shipping and the {item} came almost a week behind schedule.",
        "Needed the {item} for an event last weekend, it got here on Wednesday after.",
    ],
    "changed_mind": [
        "Honestly I just regret buying the {item}, it's not for me.",
        "My sister already got me a {item} so I've got two now.",
        "Turns out the {item} won't fit in my apartment, would like to send it back.",
        "Never opened the {item}, decided to go a different direction.",
    ],
}
INDIRECT_RATE = 0.30


class World:
    def __init__(self, conn, rng):
        self.conn, self.rng = conn, rng
        self.next_cust, self.next_order, self.next_ref = 1000, 50000, 1

    def customer(self, age_days=None, chargebacks=0, mismatch=None, refunds_30d=None, refunds_old=None):
        r = self.rng
        cid = f"C{self.next_cust}"; self.next_cust += 1
        age = age_days if age_days is not None else r.randint(60, 1500)
        mismatch = mismatch if mismatch is not None else int(r.random() < 0.1)
        name = f"{r.choice(FIRST)} {r.choice(LAST)}"
        self.conn.execute("INSERT INTO customers VALUES (?,?,?,?,?,?)",
                          (cid, name, name.lower().replace(" ", ".") + "@example.com",
                           (NOW - timedelta(days=age)).isoformat(), chargebacks, mismatch))
        # filler order history (all inside the account's lifetime)
        n_hist = r.randint(1, 12) if age > 20 else 0
        hist_orders = []
        for _ in range(n_hist):
            d = r.randint(31, max(32, age - 1)) if age > 32 else r.randint(1, max(1, age - 1))
            hist_orders.append(self.order(cid, round(r.uniform(10, 150), 2), ordered_days_ago=d, delivered=True))
        if refunds_30d is None:
            refunds_30d = 1 if r.random() < 0.15 else 0
        if refunds_old is None:
            refunds_old = r.randint(0, 2)
        for i in range(refunds_30d):
            oid = self.order(cid, round(r.uniform(10, 120), 2), ordered_days_ago=r.randint(10, 40), delivered=True)
            self.refund(oid, cid, NOW - timedelta(days=r.randint(1, 29)))
        for oid in hist_orders[:refunds_old]:
            self.refund(oid, cid, NOW - timedelta(days=r.randint(40, 300)))
        return cid

    def order(self, cid, amount, ordered_days_ago, promise_days=5, delivered=True, late_days=0,
              status=None, signature=0, final_sale=0, item=None):
        oid = f"O{self.next_order}"; self.next_order += 1
        ordered = NOW - timedelta(days=ordered_days_ago, hours=self.rng.randint(0, 10))
        promised = ordered + timedelta(days=promise_days)
        delivered_at = promised + timedelta(days=late_days) if delivered else None
        status = status or ("delivered" if delivered else "in_transit")
        self.conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (oid, cid, item or self.rng.choice(ITEMS), amount, final_sale, ordered.isoformat(),
                           promised.isoformat(), delivered_at.isoformat() if delivered_at else None, status, signature))
        return oid

    def refund(self, oid, cid, when):
        rid = f"RH{self.next_ref:05d}"; self.next_ref += 1
        amt = self.conn.execute("SELECT amount FROM orders WHERE order_id=?", (oid,)).fetchone()[0]
        self.conn.execute("INSERT INTO refunds VALUES (?,?,?,?,?,?,?,?)",
                          (rid, oid, cid, amt, "history", when.isoformat(), "history", f"hist:{rid}"))


def low(r):
    return round(r.uniform(15, 190), 2)


def msg(r, kind, item):
    pool = INDIRECT[kind] if kind in INDIRECT and r.random() < INDIRECT_RATE else MSG[kind]
    return r.choice(pool).format(item=item.lower())


# Each builder returns (customer_id, order_id, message, expected_decision, expected_amount)
def _delivered_case(w, r, kind, days_since_delivery, amount=None, final_sale=0, late_days=0, cust=None):
    cid = cust or w.customer()
    amount = amount or low(r)
    late = late_days
    ordered_ago = days_since_delivery + 5 + late
    item = r.choice(ITEMS)
    oid = w.order(cid, amount, ordered_ago, late_days=late, final_sale=final_sale, item=item)
    return cid, oid, msg(r, kind, item), amount


def s_nd_lost(w, r):
    cid = w.customer(); amt = low(r); item = r.choice(ITEMS)
    status = r.choice(["lost", "returned_to_sender", "in_transit"])
    oid = w.order(cid, amt, r.randint(20, 35), delivered=False, status=status, item=item)
    return cid, oid, msg(r, "not_delivered", item), "approve", amt


def s_nd_nosig(w, r):
    cid = w.customer(); amt = low(r); item = r.choice(ITEMS)
    oid = w.order(cid, amt, r.randint(8, 20), status="delivered", signature=0, item=item)
    return cid, oid, msg(r, "not_delivered", item), "approve", amt


def s_nd_signed(w, r):
    cid = w.customer(); amt = low(r); item = r.choice(ITEMS)
    oid = w.order(cid, amt, r.randint(8, 20), status="delivered", signature=1, item=item)
    return cid, oid, msg(r, "not_delivered", item), "escalate", None


def s_nd_early(w, r):
    cid = w.customer(); amt = low(r); item = r.choice(ITEMS)
    oid = w.order(cid, amt, r.randint(2, 4), promise_days=r.randint(5, 7), delivered=False, status="in_transit", item=item)
    return cid, oid, msg(r, "not_delivered", item), "deny", None


def s_damaged(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "damaged", r.randint(1, 40))
    return cid, oid, m, "approve", amt


def s_damaged_final(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "damaged", r.randint(1, 40), final_sale=1)
    return cid, oid, m, "approve", amt


def s_damaged_old(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "damaged", r.randint(70, 150))
    return cid, oid, m, "deny", None


def s_wrong(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "wrong_item", r.randint(1, 40))
    return cid, oid, m, "approve", amt


def s_late(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "late_delivery", r.randint(2, 25), late_days=r.randint(3, 10))
    return cid, oid, m, "approve", round(min(0.15 * amt, 25.0), 2)


def s_late_ontime(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "late_delivery", r.randint(2, 25), late_days=r.randint(-1, 1))
    return cid, oid, m, "deny", None


def s_cm(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "changed_mind", r.randint(1, 28))
    return cid, oid, m, "approve", amt


def s_cm_final(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "changed_mind", r.randint(1, 28), final_sale=1)
    return cid, oid, m, "deny", None


def s_cm_late(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "changed_mind", r.randint(35, 90))
    return cid, oid, m, "deny", None


def s_duplicate(w, r):
    cid, oid, m, amt = _delivered_case(w, r, r.choice(["damaged", "wrong_item"]), r.randint(5, 40))
    w.refund(oid, cid, NOW - timedelta(days=r.randint(1, 4)))  # customer was already refunded a few days ago
    return cid, oid, m, "deny", None


def s_high_value(w, r):
    kind = r.choice(["damaged", "wrong_item"])
    cid, oid, m, amt = _delivered_case(w, r, kind, r.randint(1, 40), amount=round(r.uniform(250, 900), 2))
    return cid, oid, m, "escalate", None


def s_fraud(w, r):
    cust = w.customer(chargebacks=2, mismatch=1)
    cid, oid, m, amt = _delivered_case(w, r, r.choice(["damaged", "wrong_item"]), r.randint(1, 40), cust=cust)
    return cid, oid, m, "escalate", None


def s_serial(w, r):
    cust = w.customer(refunds_30d=r.randint(3, 5))
    cid, oid, m, amt = _delivered_case(w, r, r.choice(["damaged", "wrong_item"]), r.randint(1, 20), cust=cust)
    return cid, oid, m, "escalate", None


def s_new_account(w, r):
    age = r.randint(4, 12)
    cust = w.customer(age_days=age, mismatch=0, refunds_30d=0, refunds_old=0)
    item = r.choice(ITEMS); amt = round(r.uniform(60, 190), 2)
    oid = w.order(cust, amt, ordered_days_ago=age - 1, promise_days=r.randint(1, 2), item=item)
    return cust, oid, msg(r, r.choice(["damaged", "wrong_item"]), item), "escalate", None


def s_not_owner(w, r):
    owner = w.customer(); requester = w.customer()
    item = r.choice(ITEMS)
    oid = w.order(owner, low(r), r.randint(10, 30), item=item)
    return requester, oid, msg(r, "damaged", item), "escalate", None


def s_ambiguous(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "ambiguous", r.randint(1, 30))
    return cid, oid, m, "escalate", None


def s_high_value_late_cm(w, r):
    cid, oid, m, amt = _delivered_case(w, r, "changed_mind", r.randint(40, 90), amount=round(r.uniform(300, 800), 2))
    return cid, oid, m, "deny", None


SCENARIOS = [
    ("not_delivered_lost", 12, s_nd_lost),
    ("not_delivered_no_signature", 10, s_nd_nosig),
    ("not_delivered_signed", 8, s_nd_signed),
    ("not_delivered_in_window", 6, s_nd_early),
    ("damaged", 20, s_damaged),
    ("damaged_final_sale", 6, s_damaged_final),
    ("damaged_outside_window", 5, s_damaged_old),
    ("wrong_item", 16, s_wrong),
    ("late_delivery", 12, s_late),
    ("late_claim_on_time", 6, s_late_ontime),
    ("changed_mind", 14, s_cm),
    ("changed_mind_final_sale", 5, s_cm_final),
    ("changed_mind_outside_window", 7, s_cm_late),
    ("duplicate_request", 6, s_duplicate),
    ("high_value", 16, s_high_value),
    ("fraud_signals", 10, s_fraud),
    ("serial_refunder", 10, s_serial),
    ("new_account", 10, s_new_account),
    ("order_not_owned", 7, s_not_owner),
    ("ambiguous_request", 10, s_ambiguous),
    ("high_value_denial", 4, s_high_value_late_cm),
]


def generate(conn, seed=42):
    rng = random.Random(seed)
    w = World(conn, rng)
    rows = []
    for name, count, fn in SCENARIOS:
        for _ in range(count):
            rows.append((name, *fn(w, rng)))
    rng.shuffle(rows)
    for i, (scenario, cid, oid, message, exp, exp_amt) in enumerate(rows, 1):
        conn.execute(
            "INSERT INTO cases(case_id, customer_id, order_id, message, scenario, expected_decision, expected_amount, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (f"CASE-{i:04d}", cid, oid, message, scenario, exp, exp_amt, NOW.isoformat()),
        )
    conn.commit()
    return len(rows)
