import { ChatOpenAI } from "@langchain/openai";


// Créer le model => ChatOpenAI
const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo",
  verbose: true
})

// Appeller le model => Promise<AIMessage>
const response = await model.invoke("Raconte moi une histoire drole")

console.log(response);
