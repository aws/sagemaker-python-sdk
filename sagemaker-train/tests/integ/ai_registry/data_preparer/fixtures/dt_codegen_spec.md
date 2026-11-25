# Transformation specification

The input is a JSONL file. Every line is a JSON object with two string fields:
`prompt` and `completion`.

Keep only the records whose `completion` field is exactly the string
`neutral`. Drop every other record.

Do not modify the records that are kept — write them out with the same
`prompt` and `completion` fields and values they had in the input.

Write the surviving records to the output as JSONL, one JSON object per line.
