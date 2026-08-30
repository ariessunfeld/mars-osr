# Contributing

Bug reports and focused pull requests are welcome. Please describe the physical
or numerical behavior being changed, cite primary sources for new physics, and
include an independent analytical or numerical cross-check in the permanent test
suite.

Before opening a pull request:

```bash
python scripts/fetch_kernels.py
pytest
```

Mark any test whose individual runtime is 30 seconds or longer with
`@pytest.mark.slow`. Do not commit downloaded kernels, generated outputs, or
large data products.
