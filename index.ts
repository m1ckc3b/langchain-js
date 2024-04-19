import { ChatOpenAI } from "langchain/chat_models/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ChatPromptTemplate, PromptTemplate, StringPromptValue } from "langchain/prompts";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";
import { RunnablePassthrough, RunnableSequence } from "langchain/runnables";
import { StringOutputParser } from "langchain/schema/output_parser";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

/** INDEXING
 * Index makes documents easy to retrieve
 */

// Load Documents
const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: "div.post-content",
  }
);
const docs = await loader.load();

// Split
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splits = await splitter.splitDocuments(docs);

// Embed
const vectorStore = await MemoryVectorStore.fromDocuments(
  splits,
  new OpenAIEmbeddings()
);


/** RETRIEVAL and GENERATION
 * Retrieval powered via similarity search
 */

// LLM
const llm = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-16k",
  temperature: 0
});

// Retriever
const retriever = MultiQueryRetriever.fromLLM({
  llm,
  retriever: vectorStore.asRetriever()
})



// Prompt -> Multi Query : Different perspectives
const promptPerspectives = PromptTemplate.fromTemplate(`
You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries related to: {question}
Output (4 queries):
`)

const generateQueries = promptPerspectives
  .pipe(new ChatOpenAI({
  modelName: "gpt-3.5-turbo-16k",
  temperature: 0
  }))
  .pipe(new StringOutputParser())
  .pipe((x) => x.split("\n"))

function composeDocs(docs) {
  return docs.map(doc => doc.pageContent).join("\n\n")
}

const retrievalChain = generateQueries.pipe(retriever).pipe(composeDocs)

// RAG
const prompt = PromptTemplate.fromTemplate(`
  Answer the following question based on this context: {context}
  Question: {question}
`)

const chain = RunnableSequence.from([
  {
    context: retrievalChain,
    question: (input) => input.question
  },
  prompt,
  llm,
  new StringOutputParser()
])

const result = chain.invoke({ question: "What is task decomposition for LLM agents?" })