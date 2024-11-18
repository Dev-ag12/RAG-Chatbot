"use client"
import Image from "next/image"
import Logo from "./assets/download.jpeg"
import {useChat} from "ai/react"
import  {Message} from "ai"
import Bubble from "./components/Bubble"
import LoadingBubble from "./components/LoadingBubble"
import PromptSuggestionsRow from "./components/PromptSuggestionsRow"

const Home=()=>{
    const {append,isLoading,messages,input,handleInputChange,handleSubmit}=useChat()

    const noMessages=!messages||messages.length===0
    const handlePrompt=(promptText)=>{
        const msg:Message={
            id:crypto.randomUUID(),
            content:promptText,
            role:"user"

        }
        append(msg)
    }
    return (
        <main>
            <Image src={Logo} width="250" alt="Logo"/>
            <section className={noMessages? "":"populated"}>
                {noMessages?(
                    <>
                        <p className="starter-text">
                            The ultimate place for Forumala One super fans!
                            Ask F1Gpt anything about F1 racing and it wil come back
                            with the most up-to-date answers.
                            we hope you enjoyy!!
                        </p>
                        <br/>
                        <PromptSuggestionsRow onPromptClick={handlePrompt}/>
                    </>
                ):(
                    <>
                    {messages.map((message,index)=><Bubble key={`message-${index}`} message={message}/>)}
                    {isLoading && <LoadingBubble/>}
                    </>
                )}
            </section>
            <form onSubmit={handleSubmit}>
                    <input className="question-box" onChange={handleInputChange} value={input} placeholder="Ask me Something.."/>
                    <input type="submit"/>
            </form>
        </main>
    )
}
export default Home
