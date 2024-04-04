# Tweaking AOAI Accelerator - Workshop

1. **[Accelerator Selection](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#accelerator-selection)**

2. **[Ingestion Optimization](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#ingestion-optimization)**
   - 2.1. [Evaluations](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#evaluations)
   - 2.2. [Ingestion Strategy](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#ingestion-strategy)
     - 2.2.1. [Layout](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#layout)
     - 2.2.2. [Read](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#read)

3. **[Chunking Strategy](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#chunking-strategy)**

4. **[Optimize Retrieval](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#optimize-retrieval)**

5. **[Data Ingestion Process Troubleshooting](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#data-ingestion-process-troubleshooting)**
   - 5.1. [Ingestion: Ensuring Accurate Data Ingestion](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#ingestion-ensuring-accurate-data-ingestion)
   - 5.2. [Rate Limit](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#rate-limit)
   - 5.3. [Document Re-upload](https://github.com/aymenfurter/aoai-accelerator-workshop/tree/main?tab=readme-ov-file#document-re-upload)



## Accelerator Selection
Ensure you are utilizing the appropriate accelerator. There are various options available for different scenarios. An overview can be found [here](https://github.com/Azure-Samples/chat-with-your-data-solution-accelerator?tab=readme-ov-file#when-should-you-use-this-repo).

## Accelerator Deployment
The deployment steps are detailed [here](https://github.com/Azure-Samples/chat-with-your-data-solution-accelerator/tree/main?tab=readme-ov-file#deploy-instructions). Alternatively, you can use the `azd up` command to deploy the application.

# Ingestion Optimization

There are different chunking strategies available. These strategies define how the document is split up and indexed into the Vector database (Azure AI Search). 

## Evaluations
<img src="https://github.com/aymenfurter/aoai-accelerator-workshop/blob/main/compare.png?raw=true">
Before we start doing any changes, I recommend performing a baseline test using [@pamelafox](https://www.github.com/pamelafox)'s [https://github.com/Azure-Samples/ai-rag-chat-evaluator](https://github.com/aymenfurter/ai-rag-chat-evaluator). 


My forked version allows pointing the evaluator repository to the accelerator. The following changes are required to make the evaluator work with the accelerator:

```diff
--- scripts/generate.py
+++ scripts/generate.py
@@ -34,7 +34,7 @@
     for doc in r:
         if len(qa) > num_questions_total:
             break
-        logger.info("Processing search document %s", doc["sourcepage"])
+        logger.info("Processing search document %s", doc["title"])
         text = doc["content"]
 
@@ -44,7 +44,7 @@
         )
 
-            citation = f"[{doc['sourcepage']}]"
+            citation = f"[{doc['title']}]"
             qa.append({"question": question, "truth": answer + citation})

--- scripts/evaluate.py
+++ scripts/evaluate.py
@@ -2,6 +2,7 @@
 import logging
 import time
+import json
 
@@ -14,11 +15,28 @@
 
+    # generate a random conversation ID
+    random_guid = "00000000-0000-0000-0000-000000000000"
+    random_guid = random_guid[:15] + str(int(time.time() * 1000))
+
-        "messages": [{"content": question, "role": "user"}],
-        "stream": False,
-        "context": parameters,
+        "conversation_id": random_guid,
+        "messages": [{"content": question, "role": "user"}]
 
@@ -28,9 +31,24 @@
-            answer = response_dict["choices"][0]["message"]["content"]
-            data_points = response_dict["choices"][0]["context"]["data_points"]["text"]
-            context = "\n\n".join(data_points)
+            answer = response_dict["choices"][0]["messages"][1]["content"]
+            data_points = response_dict["choices"][0]["messages"][0]["content"]
+            data_points_json = data_points
+            data_points_dict = json.loads(data_points_json)
+            formatted_output = ""
+
+            for citation in data_points_dict["citations"]:
+                filename = citation["title"]
+                content = citation["content"]
+                
+                content_cleaned = " ".join(content.split())
+                
+                formatted_output += f"{filename}:{content_cleaned}\n\n"
+
+            if formatted_output.endswith("\n\n"):
+                formatted_output = formatted_output[:-2]
+
+            context = formatted_output
```
**Reference Scripts:**
- [`generate.py`](https://github.com/Azure-Samples/ai-rag-chat-evaluator/blob/main/scripts/generate.py)
- [`evaluate.py`](https://github.com/Azure-Samples/ai-rag-chat-evaluator/blob/main/scripts/evaluate.py)

The `ai-rag-chat-evaluator` project provides detailed insights into performing validations, emphasizing the importance of clearly defining the questions that your AI application needs to address. For more information, refer to the [relevant section in the README](https://github.com/Azure-Samples/ai-rag-chat-evaluator?tab=readme-ov-file#generating-ground-truth-data).

## Ingestion Strategy
> [!NOTE]
> Markdown output is not yet implemented in the accelerator! There is currently logic that "translates" the output to HTML. See the implementation [here](https://github.com/Azure-Samples/chat-with-your-data-solution-accelerator/blob/9b182ab1ab0c94734075e325114e33fe46058052/code/backend/batch/utilities/helpers/AzureFormRecognizerHelper.py#L43). For more details on output to markdown format, check out the [official documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-layout?view=doc-intel-4.0.0#output-to-markdown-format).

With a baseline evaluation established, we can experiment with modifications to assess whether performance improves or declines.

### Layout
A key area of focus should be the document processing configuration, particularly the loading strategies. Various chunking strategies are available, with the default being **"layout"**, which leverages Document Intelligence to convert the layout of ingested documents into HTML code. For more details, see: [Concept of Layout](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-layout?view=doc-intel-4.0.0).

**Sample data**
```html
:unselected: <p>Microsoft</p> <p>Surface</p> <p>Surface Book User Guide With Windows 10</p> <p>Published: September 2016 Version 2.0</p> c 2016 Microsoft <p>Microsoft Surface</p> <p>Surface</p> <p>2016 Microsoft. All rights reserved.</p> <p>Blue Track Technology, Microsoft, OneNote, Outlook, PowerPoint, OneDrive, Windows, Xbox, and Xbox Live are registered trademarks of Microsoft Corporation.</p> <p>Surface and Skype are trademarks of Microsoft Corporation.</p> <p>Bluetooth is a registered trademark of Bluetooth SIG, Inc.</p> <p>Dolby and the double-D symbol are registered trademarks of Dolby Laboratories.</p> <p>This document is provided "as-is." Information in this document, including URL and other Internet website references, may change without notice.</p> c 2016 Microsoft Page ii <p>Microsoft Surface Contents</p> <p>Surface</p> <table><tr><td>Meet Surface Book</td><td>1</td></tr><tr><td>SURFACE BOOK FEATURES.</td><td>1</td></tr><tr><td>Set up your Surface Book</td><td>4</td></tr><tr><td>CHARGE YOUR SURFACE BOOK</td><td>4</td></tr><tr><td>SET UP WINDOWS</td><td>5 :unselected:</td></tr><tr><td>CHOOSE A MODE TO WORK IN</td><td>5 :unselected:</td></tr><tr><td>The basics</td><td>7</td></tr><tr><td>POWER AND CHARGING</td><td>7</td></tr><tr><td>Check the battery level.</td><td>7 :selected:</td></tr><tr><td>Making your battery last.</td><td>8</td></tr><tr><td>POWER STATES: ON, OFF, SLEEP, AND RESTART</td><td>8</td></tr><tr><td>TOUCH, KEYBOARD, PEN, AND MOUSE.</td><td>9</td></tr><tr><td>Touchscreen</td><td>9</td></tr><tr><td>Keyboard .</td><td>9</td></tr><tr><td>Touchpad or mouse.</td><td>10</td></tr><tr><td>Surface Pen (Surface Pro 4 version)</td><td>11</td></tr><tr><td>Accounts and signing in</td><td>11</td></tr><tr><td>FIRST ACCOUNT ON YOUR SURFACE</td><td>11</td></tr><tr><td>SIGN IN TO YOUR SURFACE.</td><td>11</td></tr><tr><td>USE WINDOWS HELLO</td><td>12</td></tr><tr><td>CHANGE YOUR PASSWORD, PICTURE, AND OTHER ACCOUNT SETTINGS</td><td>12</td></tr><tr><td>ADD AN ACCOUNT TO YOUR SURFACE</td><td>12</td></tr><tr><td>CREATE A CHILD ACCOUNT ON YOUR SURFACE.</td><
```

### Read
An alternative option is **"read"**, utilizing the document intelligence's "read model" for a more straightforward indexing approach with minimal formatting information, similar to what users might expect from the Playground. For additional information, consult: [Concept of Read](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-read).

**Sample data**
```html
<p>Microsoft</p> <p>Surface</p> <p>Surface Book User Guide With Windows 10</p> <p>Published: September 2016 Version 2.0</p> <p>c 2016 Microsoft <p>Microsoft Surface</p> <p>Surface</p> <p>2016 Microsoft. All rights reserved.</p> <p>Blue Track Technology, Microsoft, OneNote, Outlook, PowerPoint, OneDrive, Windows, Xbox, and Xbox Live are registered trademarks of Microsoft Corporation.</p> <p>Surface and Skype are trademarks of Microsoft Corporation.</p> <p>Bluetooth is a registered trademark of Bluetooth SIG, Inc.</p> <p>Dolby and the double-D symbol are registered trademarks of Dolby Laboratories.</p> <p>This document is provided "as-is." Information in this document, including URL and other Internet website references, may change without notice.</p> <p>c 2016 Microsoft</p> <p>Page ii <p>Microsoft Surface Contents</p> <p>Surface</p> <p>Meet Surface Book</p> <p>1</p> <p>SURFACE BOOK FEATURES. 1</p> <p>Set up your Surface Book</p> <p>4</p> <p>CHARGE YOUR SURFACE BOOK</p> <p>4</p> <p>SET UP WINDOWS</p> <p>5</p> <p>CHOOSE A MODE TO WORK IN</p> <p>5</p> <p>The basics</p> <p>7</p> <p>POWER AND CHARGING</p> <p>7</p> <p>Check the battery level.</p> <p>7</p> <p>Making your battery last.</p> <p>8</p> <p>POWER STATES: ON, OFF, SLEEP, AND RESTART</p> <p>8</p> <p>TOUCH, KEYBOARD, PEN, AND MOUSE.</p> <p>9</p> <p>Touchscreen</p> <p>9</p> <p>Keyboard .</p> <p>9</p> <p>Touchpad or mouse.</p> <p>10</p> <p>Surface Pen (Surface Pro 4 version)</p> <p>11</p> <p>Accounts and signing in 11</p> <p>FIRST ACCOUNT ON YOUR SURFACE</p> <p>11</p> <p>SIGN IN TO YOUR SURFACE.</p> <p>11</p> <p>USE WINDOWS HELLO</p> <p>12</p> <p>CHANGE YOUR PASSWORD, PICTURE, AND OTHER ACCOUNT SETTINGS</p> <p>12</p> <p>ADD AN ACCOUNT TO YOUR SURFACE</p> <p>12</p> <p>CREATE A CHILD ACCOUNT ON YOUR SURFACE.</p> <p>13</p> <p>Get to know Windows 10</p> <p>
```

As you can observe, this text contains numerous `<p>` tags. These can be effortlessly removed by eliminating the corresponding line in the `AzureFormRecognizerHelper.py`. For direct access to the relevant file, you can visit [AzureFormRecognizerHelper.py](https://github.com/Azure-Samples/chat-with-your-data-solution-accelerator/blob/main/code/backend/batch/utilities/helpers/AzureFormRecognizerHelper.py).

```diff
    form_recognizer_role_to_html = {
        "title": "h1",
        "sectionHeading": "h2",
        "pageHeader": None,
        "pageFooter": None,
-        "paragraph": "p",
    }
 ```

**Sample data**
```html
Microsoft Surface Surface Book User Guide With Windows 10 Published: September 2016 Version 2.0 c 2016 Microsoft Microsoft Surface Surface 2016 Microsoft. All rights reserved. Blue Track Technology, Microsoft, OneNote, Outlook, PowerPoint, OneDrive, Windows, Xbox, and Xbox Live are registered trademarks of Microsoft Corporation. Surface and Skype are trademarks of Microsoft Corporation. Bluetooth is a registered trademark of Bluetooth SIG, Inc. Dolby and the double-D symbol are registered trademarks of Dolby Laboratories. This document is provided "as-is." Information in this document, including URL and other Internet website references, may change without notice. c 2016 Microsoft Page ii Microsoft Surface Contents Surface Meet Surface Book 1 SURFACE BOOK FEATURES. 1 Set up your Surface Book 4 CHARGE YOUR SURFACE BOOK 4 SET UP WINDOWS 5 CHOOSE A MODE TO WORK IN 5 The basics 7 POWER AND CHARGING 7 Check the battery level. 7 Making your battery last. 8 POWER STATES: ON, OFF, SLEEP, AND RESTART 8 TOUCH, KEYBOARD, PEN, AND MOUSE. 9 Touchscreen 9 Keyboard . 9 Touchpad or mouse. 10 Surface Pen (Surface Pro 4 version) 11 Accounts and signing in 11 FIRST ACCOUNT ON YOUR SURFACE 11 SIGN IN TO YOUR SURFACE. 11 USE WINDOWS HELLO 12 CHANGE YOUR PASSWORD, PICTURE, AND OTHER ACCOUNT SETTINGS 12 ADD AN ACCOUNT TO YOUR SURFACE 12 CREATE A CHILD ACCOUNT ON YOUR SURFACE. 13 Get to know Windows 10
```

## Chunking Strategy
Having tested various ingestion strategies for our data, the next aspect to refine is the chunking strategy. Our options include:

- `layout`
- `page`
- `fixed_size_overlap`

`Layout` and `Page` utilize [langchain's markdown splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/markdown_header_metadata/) to semantically split the content. Meanwhile, `fixed_size_overlap` employs a static approach for chunking content, akin to the [Text Split skill](https://learn.microsoft.com/en-us/azure/search/cognitive-search-skill-textsplit).

The size of the chunks also plays a critical role. Larger chunks provide a broader context for the answers, whereas smaller chunks enhance precision in retrieval. Different configurations may work better for different types of data.

Feel free to experiment with these options, execute evaluations, and compare the outcomes.

# Optimize Retrieval
Now that we have tested out different options on the ingestion, we can also work on the retrieval part. First of all, the accelerator doesn't use semantic ranker. (See: https://learn.microsoft.com/en-us/azure/search/semantic-how-to-configure?tabs=portal)
We can easily add semantic ranker by altering the QuestionAnswerTool (see: https://github.com/Azure-Samples/chat-with-your-data-solution-accelerator/blob/63810762d5752e7c25967f5e922fbed51c5fc442/code/backend/batch/utilities/tools/QuestionAnswerTool.py#L32)

Here is an example how to implement semantic hybrid search
```
        # Retrieve documents as sources
        sources = self.vector_store.semantic_hybrid_search(
            query=question, k=40, semantic_configuration_name="default")

        # get first 10 sources
        sources = sources[:10]

        # Generate answer from sources
        answer_generator = LLMChain(
            llm=llm_helper.get_llm(), prompt=answering_prompt, verbose=self.verbose
        )
        sources_text = "\n\n".join(
            [f"[doc{i+1}]: {source.page_content}" for i, source in enumerate(sources)]
        )

        with get_openai_callback() as cb:
            result = answer_generator({"question": question, "sources": sources_text})

        answer = result["text"]
        print(f"Answer: {answer}")
```

If our quality still doesn't improve sufficiently, we might consider exploring more expensive retrieval optimization approaches such as [Hypothetical Document Embeddings](https://python.langchain.com/docs/use_cases/query_analysis/techniques/hyde/) or [Query Expansion](https://python.langchain.com/docs/use_cases/query_analysis/techniques/expansion/).

Here's a basic example showcasing Query Expansion:

```python
    def answer_question(self, question: str, chat_history: List[dict], **kwargs: dict):
        config = ConfigHelper.get_active_config_or_default()
        answering_prompt = PromptTemplate(
            template=config.prompts.answering_prompt,
            input_variables=["question", "sources"],
        )

        query_expansion_prompt = """You are an expert at converting user questions into database queries. \
You have access to a database of surface laptop documentation. \

Perform query expansion. If there are multiple common ways of phrasing a user question \
or common synonyms for key words in the question, make sure to return multiple versions \
of the query with the different phrasings.

If there are acronyms or words you are not familiar with, do not try to rephrase them.

Question: {question}

Return at least 3 versions of the question. (comma separated)"""

        query_expansion_prompt = PromptTemplate(
            template=query_expansion_prompt, input_variables=["question"]
        )

        query_generator = LLMChain(
            llm=LLMHelper().get_llm(), prompt=query_expansion_prompt, verbose=self.verbose
        )

        questions = query_generator({"question": question})["text"].split(",")
        print(f"Questions: {questions}")

        with get_openai_callback() as cb:
        sources = []

        sources += self.vector_store.semantic_hybrid_search(
            query=question, k=2, semantic_configuration_name="default"
        )

        for q in questions:
            sources += self.vector_store.semantic_hybrid_search(
                query=q, k=2, semantic_configuration_name="default"
            )

        llm_helper = LLMHelper()

        # Generate answer from sources
        answer_generator = LLMChain(
            llm=llm_helper.get_llm(), prompt=answering_prompt, verbose=self.verbose
        )
        sources_text = "\n\n".join(
            [f"[doc{i+1}]: {source.page_content}" for i, source in enumerate(sources)]
        )

        with get_openai_callback() as cb:
            result = answer_generator({"question": question, "sources": sources_text})

        answer = result["text"]
        print(f"Answer: {answer}")
```

Depending on the size of the index, the type of data, and latency requirements, different techniques may work. Again, we can test out different approaches and compare them using the evaluator: [https://github.com/Azure-Samples/ai-rag-chat-evaluator](https://github.com/aymenfurter/ai-rag-chat-evaluator)


# Data Ingestion Process Troubleshooting

## Ingestion: Ensuring Accurate Data Ingestion

During the ingestion process, regularly check the Application Insights "Failures Tab". It's possible that some documents may not be processed correctly during Document Intelligence. If encountered, the error message "The file is corrupted or the format is unsupported. Refer to the documentation for the list of supported formats." will be displayed.


```python
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/opentelemetry/trace/__init__.py", line 573, in use_span
    yield span
  File "/usr/local/lib/python3.11/site-packages/azure/core/tracing/decorator.py", line 89, in wrapper_use_tracer
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/azure/ai/formrecognizer/_document_analysis_client.py", line 198, in begin_analyze_document_from_url
    return _client_op_path.begin_analyze_document(  # type: ignore
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/azure/core/tracing/decorator.py", line 89, in wrapper_use_tracer
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/azure/ai/formrecognizer/_generated/v2023_07_31/operations/_document_models_operations.py", line 518, in begin_analyze_document
    raw_result = self._analyze_document_initial(  # type: ignore
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/azure/ai/formrecognizer/_generated/v2023_07_31/operations/_document_models_operations.py", line 443, in _analyze_document_initial
    raise HttpResponseError(response=response)
azure.core.exceptions.HttpResponseError: (InvalidRequest) Invalid request.
Code: InvalidRequest
Message: Invalid request.
Inner error: {
    "code": "InvalidContent",
    "message": "The file is corrupted or format is unsupported. Refer to documentation for the list of supported formats."
}
```

## Rate Limit

Ensure you're not encountering any rate limits. By default, when deploying with the Azure Developer CLI, the Embedding endpoint might not be scaled to its maximum capacity. An improperly configured TPM on the Embedding endpoint can lead to extended indexing times or partially indexed documents:
```

Eception while executing function: Functions.batch_push_results Result: Failure
Exception: RateLimitError: Error code: 429 - {'error': {'code': '429', 'message': 'Requests to the Embeddings_Create Operation under Azure OpenAI API version 2023-10-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please retry after 9 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.'}}
Stack:   File "/azure-functions-host/workers/python/3.11/LINUX/X64/azure_functions_worker/dispatcher.py", line 505, in _handle__invocation_request
    call_result = await self._loop.run_in_executor(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/azure-functions-host/workers/python/3.11/LINUX/X64/azure_functions_worker/dispatcher.py", line 778, in _run_sync_func
    return ExtensionManager.get_sync_invocation_wrapper(context,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/azure-functions-host/workers/python/3.11/LINUX/X64/azure_functions_worker/extension.py", line 215, in _raw_invocation_wrapper
    result = function(**args)
             ^^^^^^^^^^^^^^^^
  File "/home/site/wwwroot/BatchPushResults.py", line 49, in batch_push_results
    document_processor.process(source_url=file_sas, processors=processors)
  File "/home/site/wwwroot/utilities/helpers/DocumentProcessorHelper.py", line 41, in process
    raise e
  File "/home/site/wwwroot/utilities/helpers/DocumentProcessorHelper.py", line 38, in process
    return vector_store.add_documents(documents=documents, keys=keys)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/langchain_core/vectorstores.py", line 138, in add_documents
    return self.add_texts(texts, metadatas, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/langchain_community/vectorstores/azuresearch.py", line 309, in add_texts
    embeddings = [self.embedding_function(x) for x in texts]
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/langchain_community/vectorstores/azuresearch.py", line 309, in <listcomp>
    embeddings = [self.embedding_function(x) for x in texts]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/langchain_openai/embeddings/base.py", line 546, in embed_query
    return self.embed_documents([text])[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/langchain_openai/embeddings/base.py", line 517, in embed_documents
    return self._get_len_safe_embeddings(texts, engine=engine)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/langchain_openai/embeddings/base.py", line 333, in _get_len_safe_embeddings
    response = self.client.create(
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/resources/embeddings.py", line 113, in create
    return self._post(
           ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1208, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 897, in request
    return self._request(
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 973, in _request
    return self._retry_request(
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1021, in _retry_request
    return self._request(
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 973, in _request
    return self._retry_request(
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1021, in _retry_request
    return self._request(
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 988, in _request
    raise self._make_status_error_from_response(err.response) from None
```

## Document Re-upload

During my testing, I experienced documents not properly deleted and recreated. I recommend deleting the files both from the storage account and the Admin GUI to ensure no old documents are reindexed.
