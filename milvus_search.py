def search_text_and_images(query, top_k=3):
    """
    Temporary mock search function.
    Replace this with real Milvus search later.
    """

    return [
        {
            "type": "text",
            "content": "This is a sample retrieved manual section related to wiring diagrams."
        },
        {
            "type": "image",
            "image": "wiring_diagram_page3.png"
        }
    ]
