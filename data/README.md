# Dataset notes (payloads are git-ignored)

Raw cart-event payloads live in `data/raw/` and are git-ignored. Commit
schemas and *synthetic* sample generators only.

Expected event frame (CSV):

| column     | type        | meaning                         |
|------------|-------------|---------------------------------|
| `user_id`  | int/category| client (cart owner)             |
| `item_id`  | int/category| product placed in cart          |
| `event_ts` | timestamp   | cart event time (optional)      |
| `qty`      | int         | quantity (optional, default 1)  |

Run `python -m experiments.run_fed` to generate an in-memory synthetic dataset
if no payload is present.