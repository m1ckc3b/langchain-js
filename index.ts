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
