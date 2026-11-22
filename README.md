# moby-mirror-jobs

Jobs that copy **public, openly licensed data** into a public Hugging Face dataset repo as a fallback copy for a study site.

- `scripts/pmc_shard.py` + `.github/workflows/pmc-shards.yml`: copies the commercial-use subset (CC0, CC BY, CC BY-SA, CC BY-ND) of the PubMed Central Open Access dataset (from the PMC Cloud Service on AWS) in 20 parallel shards. Each article's XML is checked against the md5 PMC publishes. Nothing private is read or written. The Hugging Face write token is the repository secret `HF_TOKEN`.
