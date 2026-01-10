import requests

# url = f"http://192.168.31.234:3000"
url = f"http://192.168.31.176:3000"

group_id = 792573256


def push_message(message: str):
    print(f"push message: {message}")
    # return
    try:
        requests.post(
            f"{url}/send_group_msg",
            json={
                "group_id": group_id,
                "message": [
                    {
                        "type": "text",
                        "data": {"text": f"{message}"},
                    }
                ],
            },
        )
    except Exception as ex:
        print(ex)
