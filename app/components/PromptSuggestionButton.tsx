const PromptSuggestionButton=({text,onCLick})=>{
    return (
        <button className="prompt-suggestion-button"
            onClick={onCLick}
        >
            {text}
        </button>
    )
}

export default PromptSuggestionButton