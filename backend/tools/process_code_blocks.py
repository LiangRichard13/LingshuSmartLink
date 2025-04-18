import re
def check_and_process_code_blocks(text:str, action:str="extract")->str:
    """
    Check for ```markdown``` or ```json``` blocks and process them.
    
    Args:
        text (str): Input text.
        action (str): "remove", "extract", or "replace".

        remove: 去除代码块
        extract: 提取代码块
        replace: 替换代码块为[CODE BLOCK]
    
    Raises:
        ValueError: If action is not 'remove', 'extract', or 'repla
    
    Returns:
        str: Processed text.
    """
    pattern = r'```(markdown|json)([\s\S]*?)```'
    
    if not re.search(pattern, text):
        return text  # No code blocks found
    
    if action == "remove":
        return re.sub(pattern, '', text)
    elif action == "extract":
        return re.sub(pattern, r'\2', text)
    elif action == "replace":
        return re.sub(pattern, '[CODE BLOCK]', text)
    else:
        raise ValueError("Invalid action. Use 'remove', 'extract', or 'replace'.")