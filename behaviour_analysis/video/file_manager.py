def video_code_to_timestamp(video_code):
    timestamp = video_code[-6:]
    timestamp = '-'.join([timestamp[i:i+2] for i in range(0, 6, 2)])
    return timestamp
