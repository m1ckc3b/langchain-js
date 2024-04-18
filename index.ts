import { ChatOpenAI } from "langchain/chat_models/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PromptTemplate } from "langchain/prompts";
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
  docs,
  new OpenAIEmbeddings()
);
const retriever = vectorStore.asRetriever({});

/** RETRIEVAL and GENERATION
 * Retrieval powered via similarity search
 */

// LLM
const llm = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-16k",
  temperature: 0,
});

// Prompt -> Multi Query : Different perspectives
const promptPerspectives = PromptTemplate.fromTemplate(`
  You are an AI language model assistant. Your task is to generate five 
  different versions of the given user question to retrieve relevant documents from a vector 
  database. By generating multiple perspectives on the user question, your goal is to help
  the user overcome some of the limitations of the distance-based similarity search. 
  Provide these alternative questions separated by newlines. Original question: {question}
`);

const prompt = PromptTemplate.fromTemplate(`
  Answer the following question based on this context: {context}
  Question: {question}
`);

function getUniqueUnion(documents: Document[][]): Document[] {
  // Flatten list of lists and convert each Document to string
  const flattenedDocs: string[] = documents.flatMap((sublist) =>
    sublist.map((doc) => JSON.stringify(doc))
  );

  // Get unique documents using Set for efficient deduplication
  const uniqueDocsSet = new Set(flattenedDocs);

  // Return the unique documents by parsing back from JSON strings
  return Array.from(uniqueDocsSet).map((docString) => JSON.parse(docString));
}

const chainPerspectives = promptPerspectives
  .pipe(llm)
  .pipe(new StringOutputParser())
  .pipe((x) => x.split("\n"));
const retrievalChain = chainPerspectives
  .pipe(retriever.map())
  .pipe(getUniqueUnion);

const finalChain = RunnableSequence.from([
  {
    context: retrievalChain,
    question: new RunnablePassthrough(),
  },
  prompt,
  llm,
  new StringOutputParser(),
]);

const response = await finalChain.invoke({
  question: "What is task decomposition for LLM agents?"
})
console.log(response);