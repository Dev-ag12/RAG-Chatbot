// import { GoogleGenerativeAI } from "@google/generative-ai"
// import { streamText, StreamData } from "ai";
// import { DataAPIClient } from "@datastax/astra-db-ts"

// const ASTRA_DB_NAMESPACE = process.env.ASTRA_DB_NAMESPACE as string;
// const ASTRA_DB_COLLECTION = process.env.ASTRA_DB_COLLECTION as string;
// const ASTRA_DB_ENDPOINT = process.env.ASTRA_DB_ENDPOINT as string;
// const ASTRA_DB_APPLICATION_TOKEN = process.env.ASTRA_DB_APPLICATION_TOKEN as string;
// const GEMINI_API_KEY = process.env.GEMINI_API_KEY as string;

// const genAI = new GoogleGenerativeAI(GEMINI_API_KEY)
// const client=new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
// const db=client.db(ASTRA_DB_ENDPOINT,{namespace:ASTRA_DB_NAMESPACE})

// export async function POST(req:Request){
//     try{
//         const data = new StreamData();
//         const {messages}=await req.json()
//         const latestMessage=messages[messages?.length-1]?.content
//         data.append({ status: "Processing request" });

//         let docContext=""
//         const model = genAI.getGenerativeModel({ model: "text-embedding-004"})
//         const result = await model.embedContent(latestMessage);

//         try{
//             const collection=await db.collection(ASTRA_DB_COLLECTION)
//             const cursor=collection.find(null,{
//                 sort:{
//                     $vector:result.embedding.values
//                 },
//                 limit:10
//             })

//             const documents=await cursor.toArray()

//             const docsMap=documents?.map(doc=>doc.text)
//             docContext=JSON.stringify(docsMap)
//             data.append({ status: "Database query successful", docsMap });
//         }catch(err){
//             console.log("Error querying db..")
//         }
        
//         const template = {
//             role: "system",
//             content: `You are an AI assistant who knows everything about Formula One. 
//                 Use the below context to augment what you know about Formula One racing. 
//                 The context will provide you with the most recent page data from Wikipedia, 
//                 the official F1 website and others. 
//                 If the context doesn't include the information you need, answer based on your 
//                 existing knowledge and don't mention the source of your information or 
//                 what the context does or doesn't include. 
//                 Format responses using markdown where applicable and don't return images.
//                 ------------------
//                 START CONTEXT
//                 ${docContext}
//                 END CONTEXT
//                 ---------------------`
//         }
//         const response =await openai.chat.completions.create({
//             model:"gpt-4",
//             stream:true,
//             messages:[template, ...messages]
//         })

//         const stream=OpenAIStream(response)
//         return new streamText.toDataStreamResponse(stream) 

      
//     }catch(err){
//         throw err
//     }
// }