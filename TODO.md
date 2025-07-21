## GraphRAG pipeline automation build & front end

- create a new branch for the changes below
- create "build your own graph" pipeline that takes users PDFs, suggests list of entities to create the graph with. creates and suggests a list of benchmark questions and answers. then i guess we add them as a variable and refactor the current graphprocessor to take in the entities as those variables?
-- create a simple front end and onboarding journey
-- check all files if they need generalisating (graph and advanced graph processor. Text2cypher retriever, graphrag and advanced graphrag retriever, etc.)
-- when the tests are done, push the images to the front end for a nice report.

## other changes
- Understand how microsoft graphrag team benchmark, it looks like the provide 1-2-4 scores rather than our method.
- add the simple examples from our github to each retriever.
- throw in the ms_graphrag notebook and get claude to check the differences and no steps missed (notebook uses gds for community detection but not in our pipeline)
- once repo is working well, add a custom pipeline, schema and query engine to see how much more performance you gain vs my generic approach https://medium.com/data-science/building-knowledge-graphs-with-llm-graph-transformer-a91045c49b59

# improve benchmark metrics
-- create a much better list of questions.
-- use microsoft paper system prompts to retriever and question set generation. (page 21)