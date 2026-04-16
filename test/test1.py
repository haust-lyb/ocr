from paddlex import create_model

model = create_model("PP-DocLayoutV3")

results = model.predict("test1.png")

for result in results:
    result.print()
    result.save_to_img("test1_out.png")
