import {DataAPIClient} from "@datastax/astra-db-ts"
import {PuppeteerWebBaseLoader} from "langchain/document_loaders/web/puppeteer"
import { GoogleGenerativeAI } from "@google/generative-ai"

import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"

import "dotenv/config"

type SimilarityMetric="dot_product"| "cosine"|"euclidean"

const ASTRA_DB_NAMESPACE = process.env.ASTRA_DB_NAMESPACE as string;
const ASTRA_DB_COLLECTION = process.env.ASTRA_DB_COLLECTION as string;
const ASTRA_DB_ENDPOINT = process.env.ASTRA_DB_ENDPOINT as string;
const ASTRA_DB_APPLICATION_TOKEN = process.env.ASTRA_DB_APPLICATION_TOKEN as string;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY as string;


const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

const f1Data=[
        'https://en.wikipedia.org/wiki/Formula_One',
        'https://www.formula1.com/en/latest/all',
        'https://www.forbes.com/sites/brettknight/2023/11/29/formula-1s-highest-paid-drivers-2023/?sh=12bdb942463f',
        'https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship',
        'https://en.wikipedia.org/wiki/2022_Formula_One_World_Championship',
        'https://en.wikipedia.org/wiki/List_of_Formula_One_World_Drivers%27_Champions',
        'https://www.formula1.com/en/results.html/2024/races.html',
        'https://www.formula1.com/en/racing/2024.html'
]

const client =new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db =client.db(ASTRA_DB_ENDPOINT,{namespace:ASTRA_DB_NAMESPACE});

const splitter=new RecursiveCharacterTextSplitter({
    chunkSize:512,
    chunkOverlap:100
})

const createCollection=async(similarityMetric: SimilarityMetric="dot_product")=>{
    const res=await db.createCollection(ASTRA_DB_COLLECTION,{
        vector:{
            dimension:768,
            metric: similarityMetric
        }
    })
    console.log(res)
}

const loadSampleData=async()=>{
    const collection=await db.collection(ASTRA_DB_COLLECTION)
    const model = genAI.getGenerativeModel({ model: "text-embedding-004"})
    for await(const url of f1Data){
        const content=await scrapePage(url)
        const chunks=await splitter.splitText(content)
        for await (const chunk of chunks){
            const result = await model.embedContent(chunk);
            const vector = result.embedding.values
            const res=await collection.insertOne({
                $vector:vector,
                text:chunk
            })
            console.log(res)
        }
    }
}

const scrapePage=async(url:string)=>{
    const loader=new PuppeteerWebBaseLoader(url,{
        launchOptions:{
            headless:true
        },
        gotoOptions:{
            waitUntil:"domcontentloaded"
        },
        evaluate:async(page,browser)=>{
            const result=await page.evaluate(()=>document.body.innerHTML)
            await browser.close()
            return result
        }
    })
    return (await loader.scrape())?.replace(/<[^>]*>?/gm, '')
}

createCollection().then(()=>loadSampleData())
