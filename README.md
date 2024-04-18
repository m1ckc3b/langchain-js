# langchain-js

## 1 - Introduction

Installation du package
```
bun add langchain @langchain/openai
```

Création d'un fichier `.env`
```
OPENAI_API_KEY=<your_key>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your_key>
```

Dans le fichier `index.ts`, on instantie un nouveau model (ou llm)
```
const model = new ChatOpenAI()
```
On peut paramétrer le modèle en passant un objet contenant les paramètres `modelName`, `temperature`, `maxTokens` ou encore `verbose`.

```
const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo",
  verbose: true
})
```

Pour interagir avec le model, il suffit de l'appeler avec la méthode invoke à qui l'on passse un prompt en argument :
```
await model.invoke(prompt)

// Exemple:
const response = await model.invoke("Raconte moi une histoire drole")
```
Cette méthode retourne une promesse de type AIMessage, très complexe. Après résolution de la promessse, il est possible de lire réponse du llm au prompt depuis la propriété `content`.

```
AIMessage {
  lc_serializable: true,
  lc_kwargs: {
    content: "Il était une fois...",
    additional_kwargs: {
      function_call: undefined,
      tool_calls: undefined,
    },
    response_metadata: {},
  },
  lc_namespace: [ "langchain_core", "messages" ],
  content: "Il était une fois...",
  name: undefined,
  additional_kwargs: {
    function_call: undefined,
    tool_calls: undefined,
  },
  response_metadata: {
    tokenUsage: {
      completionTokens: 336,
      promptTokens: 15,
      totalTokens: 351,
    },
    finish_reason: "stop",
  },
  _getType: [Function: _getType],
  lc_aliases: [Getter],
  text: [Getter],
  toDict: [Function: toDict],
  toChunk: [Function: toChunk],
  lc_id: [Getter],
  lc_secrets: [Getter],
  lc_attributes: [Getter],
  toJSON: [Function: toJSON],
  toJSONNotImplemented: [Function: toJSONNotImplemented],
}
```
