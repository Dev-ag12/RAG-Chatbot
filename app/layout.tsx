import "./global.css"

export const metadata={
    title:"F1GPT",
    description :"The place to go for formula one questions!!"
}

const RootLayout=({children})=>{
    return(
        <html lang="en">
            <body>{children}</body>
        </html>
    )
}

export default RootLayout