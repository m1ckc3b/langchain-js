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
const loader = new CheerioWebBaseLoader("url",{})
const docs = await loader.load()

// Split 
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200
})
const splits = await splitter.splitDocuments(docs)

// Embed
const vectorStore = await MemoryVectorStore.fromDocuments(
  docs,
  new OpenAIEmbeddings()
)
const retriever = vectorStore.asRetriever({})
const documents = await retriever.getRelevantDocuments("")

/** RETRIEVAL and GENERATION 
 * Retrieval powered via similarity search
*/

// Prompt
const prompt = PromptTemplate.fromTemplate(`
    Answer the question based only on the following context: {context}
    
    Question: {question}
`)

// LLM
const llm = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0
})

// Post-processing
function formatDocs(docs) {
  return docs.map(doc => doc.pageContent).join("\n\n")
}

// Chain
const ragChain = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocs),
    question: new RunnablePassthrough()
  },
  prompt,
  llm,
  new StringOutputParser()
])

// Question
ragChain.invoke("my question is...")