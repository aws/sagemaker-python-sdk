# Auto-Labelling Specification: Banking support intents (integration fixture)

## Use Case Context
Each row is one short message a customer sent to a retail bank's support chat. Assign the single intent the customer is asking about.

## Labelling Configuration
### Label Set

| Label | Definition |
|---|---|
| card_arrival | The customer ordered a card and asks where it is or how to track the delivery. |
| card_not_working | The customer's card is not working and no more specific cause is stated. |
| declined_card_payment | A card payment was declined or refused. |
| lost_or_stolen_card | The customer's card was lost or stolen, or they want to report it so. |
| top_up_failed | A top-up to the account did not complete. |

### Labelling Rules

- Pick the single intent the customer is asking about. Use the most specific label that fits; use `card_not_working` only when no more specific cause is stated.
- A declined payment is `declined_card_payment` even when the customer blames the card.
- A missing card that was never received is `card_arrival`; a card the customer had and lost is `lost_or_stolen_card`.

## Output Schema
Per record: a short reasoning chain, then the final label (one of the Label Set). Ground truth, if present in the input data, is for scoring only and must not be read when predicting.

```
{"reasoning": "<one short sentence explaining the label choice>", "label": "<one of the labels from the Label Set above>"}
```

## Data Verification

### Verification Mode
deterministic

### Checks
1. Output is a single JSON object with "reasoning" and "label" keys.
2. label is one of the labels in the Label Set (exact match, case-sensitive).
3. reasoning is non-empty.
