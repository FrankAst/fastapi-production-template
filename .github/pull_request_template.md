# Pull Request Template

## Summary

This PR includes:
- Modification of endpoint POST /train/ to accept a CSV file with at least 2 columns
where the last one must be the Target
- Endpoint data model to load and run light validations on the CSV.
- Domain PredictionInput modified to support a matrix with more than one feature.

- Created to endpoints for predicting: prediction/single & prediction/batch. The latter
supports a csv file with predictors features. 
- Created unified prediction service to serve both single & batch requests.


## Checklist

### Code & Implementation
- [x] Have you run `uv run poe format` and fixed all warnings before submitting
this PR? -> Yes, while I'm still working on fixing some minor issues I believe some will get improved after
your feedback. 
- [x] Is all the logic in services instead of endpoints? -> Yes - but perhaps it can be improved.
- [x] Is the code self-explanatory without requiring comments or documentation? -> Should be. 

### Machine Learning Specific
- [ ] Are the data preprocessing steps reproducible? (e.g., deterministic pipelines, seeds set).
- [ ] Model changes are tracked with experiment tracker (e.g., versioned
artifacts, experiment logs).
- [ ] Can all the notebooks be run top to bottom?
- [ ] Are all the notebooks outputs cleared?

### Documentation
- [ ] Relevant README or docs updated.

---

### Reviewer Guidance
- Ensure clarity: Could the reviewer understand the purpose of the PR without
asking?
- Ensure reproducibility: Could you re-run the experiment or training with the
given code and instructions?
- Ensure maintainability: Does the structure fit into the long-term system
design?
