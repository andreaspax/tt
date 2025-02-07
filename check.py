import numpy as np

# Load the test triples from the .npy file
test_triples = np.load('test_averaged_triples.npy', allow_pickle=True)

# Optionally, print the shape or a sample of the loaded triples
print(f"Loaded test triples with shape: {test_triples.shape}")
print("Sample test triple:", test_triples[0] if len(test_triples) > 0 else "No triples found")

# # print(test_triples[0][2])  # Accessing the third object of the first index
# # Convert the loaded test triples to a list for easier manipulation
# test_triples_list = test_triples.tolist()

# # Find all rows where the query matches "does human hair stop squirrels"
# matching_triples = [triple for triple in test_triples_list if triple[0] == 'how early can you renew your driving licence']

# # Print the matching triples
# print(f"Found {len(matching_triples)} matching triples:")
# for triple in matching_triples:
#     print(triple)


