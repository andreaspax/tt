import testgeneratetriplet
import pandas as pd

triple_generator = testgeneratetriplet.TripleGenerator()

test_dataset = pd.read_json("test_dataset.json", orient="records", lines=True).head(1)

test_triples = triple_generator.generate_triples(test_dataset, "Test Dataset")


print(test_triples[:5])
print(f"Shape of the generated triples: {len(test_triples)}")
