from langchain_core.tools import tool


@tool
def summarize_text(text: str) -> str:
    """
    Summarizes long passages of text into key bullet points and executive insights.
    
    Args:
        text: The text content to be summarized.
        
    Returns:
        A formatted summary outlining key points and core takeaway.
    """
    cleaned = text.strip()
    if len(cleaned) < 50:
        return f"Text is already brief:\n- {cleaned}"
        
    # Split text into sentences for basic extractive key points
    sentences = [s.strip() for s in cleaned.replace("\n", " ").split(".") if s.strip()]
    
    # Pick top representative sentences
    key_points = sentences[:3] if len(sentences) >= 3 else sentences
    bullet_list = "\n".join([f"• {point}." for point in key_points])
    
    return (
        f"Text Summary ({len(cleaned)} characters reduced to core insights):\n\n"
        f"Key Highlights:\n{bullet_list}\n\n"
        f"Overview: The text primarily addresses {sentences[0] if sentences else 'the provided topic'}."
    )
