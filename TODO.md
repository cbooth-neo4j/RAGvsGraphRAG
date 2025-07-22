# GraphRAG automated benchmarking pipeline

- create "build your own graph" pipeline that takes users PDFs, suggests list of entities to create the graph with. creates and suggests a list of benchmark questions and answers. then we add them as a variable and refactor the current graphprocessor to take in the entities as those variables.
- Merge advanced graph Processor into graph processor?
- create a simple front end and onboarding journey
- check all files if they need generalisating (graph and advanced graph processor. Text2cypher retriever, graphrag and advanced graphrag retriever, etc.)
- when the tests are done, push the chart reports to the front end for a nice report.
- ability to ask indiviudal questions and output both responses for a side by side comparison of the answers? (Xavier sounds like he's doing this)

- add the simple examples from our product examples from our github to each retriever.
- throw in the ms_graphrag notebook and get claude to check the differences and no steps missed (notebook uses gds for community detection but not in our pipeline)
- once repo is working well, add a custom pipeline, schema and query engine to see how much more performance you gain vs my generic approach https://medium.com/data-science/building-knowledge-graphs-with-llm-graph-transformer-a91045c49b59

## improve benchmark metrics
- create a much better list of template questions and dataset (open source dataset or custom?)
- use microsoft paper system prompts to retriever and question set generation. (page 21)
- Change metrics to a scoring system (like microsoft GraphRAG paper. They score 0, 2, 4 depending on quality of answer) instead of scoring 0 to 1.

## Add more retrievers
- add other neo4j python retrievers (hybrid / VectorCypherRetriever, etc)
- understand how LLM graph builder works and try to create a general template

## Improve retriever performance
- Improve graphrag retriever performance (it's not much better than vector-only atm, and factual correctness occassionally scores lower?!)
-- check that im building the graph correctly (im not sure whether the communities are building correctly according to MicrosoftGraphRAG paper and comparing to LLM Graph builder)