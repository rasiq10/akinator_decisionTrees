from __future__ import print_function
import re

# Dataset.
# Format: each row is an entry.
# The last column is the label.
# The first n columns are features.


#training_data = pd.read_excel(r"F:\Uni\CS 590-02(Expert Systems)\Akinator\1Test.xlsx")

training_data = [
    ['Yes','Yes','Actor','American','Male','Fury','Brad Pitt'],
    ['Yes','Yes','Actor','English','Male','Memory','Liam Neeson'],
    ['Yes','Yes','Actor','American','Male','Spirited','Will Ferrell'],
    ['Yes','Yes','Actor','American','Male','Sonic','Jim Carrey'],
    ['Yes','Yes','Actor','American','Male','Flight','Denzel Washington'],
    ['Yes','Yes','Actor','English','Male','Sherlock','Benedict Cumberbatch'],
    ['Yes','No','Actor','American','Male','Gatsby','Leonardo Dicaprio'],
    ['Yes','Yes','Actor','American','Male','Intern','Robert De Niro'],
    ['Yes','Yes','Actor','American','Male','Sully','Tom Hanks'],
    ['Yes','Yes','Actor','American','Male','Glass','Samuel L. Jackson'],
    ['Yes','Yes','Actor','American','Male','Wander','Tommy Lee Jones'],
    ['Yes','Yes','Actor','English','Male','Beast','Idris Elba'],
    ['Yes','No','Actor','American','Male','Transcedence','Johnny Depp'],
    ['Yes','Yes','Actor','American','Male','Lucy','Morgan Freeman'],
    ['Yes','Yes','Actor','English','Male','Spy','Jason Statham'],
    ['Yes','No','Actor','American','Male','Oblivion','Tom Cruise'],
    ['Yes','Yes','Actor','American','Male','Patriot','Heath Ledger'],
    ['Yes','Yes','Actor','English','Male','Xmen','Patrick Stewart'],
    ['Yes','Yes','Actor','English','Male','Batman','Christian Bale'],
    ['Yes','Yes','Actor','American','Male','Solaris','George Clooney'],
    ['Yes','Yes','Actor','English','Male','Interstellar','Michael Caine'],
    ['Yes','Yes','Actor','American','Male','Martian','Matt Damon'],
    ['Yes','Yes','Actor','American','Male','Fugitive','Harrison Ford'],
    ['Yes','Yes','Actor','American','Male','Terminator','Arnold Schwarzenegger'],
    ['Yes','Yes','Actor','English','Male','Legend','Tom Hardy'],
    ['No','No','Soldier','Japanese','Female','Attack on Titan','Mikasa Ackerman'],
    ['No','No','Criminal','Japanese','Male','Naruto','Itachi Uchiha'],
    ['No','No','Artist','Japanese','Male','Death Note','Misa Misa'],
    ['No','Yes','Ninja','Japanese','Male','Naruto','Naruto Uzumaki'],
    ['No','Yes','Alchemist','Japanese','Male','Fullmetal Alchemist','Edward Elric'],
    ['No','No','Killer','Japanese','Male','HunterXHunter','Killua Zoldyck'],
    ['No','No','Soldier','Japanese','Female','Steins Gate','Rintaro Okabe'],
    ['No','No','Student','Japanese','Male','Death Note','Light Yagami'],
    ['No','No','Swordsman','Japanese','Male','One Piece','Zoro'],
    ['No','No','Pirate','Japanese','Male','One Piece','Luffy'],
    ['No','No','Detective','Japanese','Male','Death Note','L Lawliet'],
    ['No','No','Soldier','Japanese','Male','Attack on Titan','Levi Ackerman'],
    ['No','No','King','Japanese','Male','Code Geass','Lelouch Lamperouge'],
    ['No','No','Superhero','American','Male','First Avenger','Captain America'],
    ['No','No','Superhero','American','Male','Dark Knight','Batman'],
    ['No','No','Superhero','American','Male','Man of Steel','Superman'],
    ['No','No','Superhero','American','Female','Dawn of Justice','Wonder Woman'],
    ['No','No','Superhero','American','Female','Justice League','Jean Grey'],
    ['No','No','Superhero','American','Female','Batman','Batgirl'],
    ['No','No','Superhero','American','Female','Ant Man','Wasp'],
    ['Yes','Yes','Actor','English','Female','Atonement','Keira Knightley'],
    ['Yes','Yes','Actor','American','Female','One Day','Anne Hathaway'],
    ['Yes','No','Actor','American','Female','Divergent','Shailene Woodley'],
    ['Yes','No','Actor','American','Female','Percy Jackson','Alexandra Daddario'],
    ['Yes','No','Actor','American','Female','Vampire Academy','Zoey Deutch']
]


# Column labels.
# These are used only for printing
header = ["Real","Married","Occupation","Nationality","Gender","Movies","Names"]


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])



def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number'  and a
    'column value'. The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question."""

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))




def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows




def gini(rows):
    """Calculate the Gini Impurity for a list of rows."""
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)


            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class - number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch




def build_tree(rows):
    """Builds the tree"""

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # depending on the answer.


    return Decision_Node(question, true_branch, false_branch)

    


def print_tree(node, spacing=""):

    #print(node.left)
    if isinstance(node, Leaf):
        temp = str(node.predictions).split('\'')[1]
        print ("Is your character", temp)
        y = input()
        if(y == "y"):
            exit
        return

    # Print the question at this node
    #print (spacing + str(node.question))
    x=input(str(node.question))

    if(x == "y"):
    #    b = False
        # Call this function recursively on the true branch
        #print (spacing + '--> True:')
        print_tree(node.true_branch, spacing + "  ")    
    if(x == "n" ):
     #   b = True
        # Call this function recursively on the false branch
        #print (spacing + '--> False:')
        print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs



if __name__ == '__main__':

    my_tree = build_tree(training_data)

    print_tree(my_tree)

    

    # Evaluate
    testing_data = [
    ['Yes','Yes','Actor','American','Male','Fury','Brad Pitt'],
    ['Yes','Yes','Actor','English','Male','Memory','Liam Neeson'],
    ['Yes','Yes','Actor','American','Male','Spirited','Will Ferrell'],
    ['Yes','Yes','Actor','American','Male','Sonic','Jim Carrey'],
    ['Yes','Yes','Actor','American','Male','Flight','Denzel Washington'],
    ['Yes','Yes','Actor','English','Male','Sherlock','Benedict Cumberbatch'],
    ['Yes','No','Actor','American','Male','Gatsby','Leonardo Dicaprio'],
    ['Yes','Yes','Actor','American','Male','Intern','Robert De Niro'],
    ['Yes','Yes','Actor','American','Male','Sully','Tom Hanks'],
    ['Yes','Yes','Actor','American','Male','Glass','Samuel L. Jackson'],
    ['Yes','Yes','Actor','American','Male','Wander','Tommy Lee Jones'],
    ['Yes','Yes','Actor','English','Male','Beast','Idris Elba'],
    ['Yes','No','Actor','American','Male','Transcedence','Johnny Depp'],
    ['Yes','Yes','Actor','American','Male','Lucy','Morgan Freeman'],
    ['Yes','Yes','Actor','English','Male','Spy','Jason Statham'],
    ['Yes','No','Actor','American','Male','Oblivion','Tom Cruise'],
    ['Yes','Yes','Actor','American','Male','Patriot','Heath Ledger'],
    ['Yes','Yes','Actor','English','Male','Xmen','Patrick Stewart'],
    ['Yes','Yes','Actor','English','Male','Batman','Christian Bale'],
    ['Yes','Yes','Actor','American','Male','Solaris','George Clooney'],
    ['Yes','Yes','Actor','English','Male','Interstellar','Michael Caine'],
    ['Yes','Yes','Actor','American','Male','Martian','Matt Damon'],
    ['Yes','Yes','Actor','American','Male','Fugitive','Harrison Ford'],
    ['Yes','Yes','Actor','American','Male','Terminator','Arnold Schwarzenegger'],
    ['Yes','Yes','Actor','English','Male','Legend','Tom Hardy'],
    ['No','No','Soldier','Japanese','Female','Attack on Titan','Mikasa Ackerman'],
    ['No','No','Criminal','Japanese','Male','Naruto','Itachi Uchiha'],
    ['No','No','Artist','Japanese','Male','Death Note','Misa Misa'],
    ['No','Yes','Ninja','Japanese','Male','Naruto','Naruto Uzumaki'],
    ['No','Yes','Alchemist','Japanese','Male','Fullmetal Alchemist','Edward Elric'],
    ['No','No','Killer','Japanese','Male','HunterXHunter','Killua Zoldyck'],
    ['No','No','Soldier','Japanese','Female','Steins Gate','Rintaro Okabe'],
    ['No','No','Student','Japanese','Male','Death Note','Light Yagami'],
    ['No','No','Swordsman','Japanese','Male','One Piece','Zoro'],
    ['No','No','Pirate','Japanese','Male','One Piece','Luffy'],
    ['No','No','Detective','Japanese','Male','Death Note','L Lawliet'],
    ['No','No','Soldier','Japanese','Male','Attack on Titan','Levi Ackerman'],
    ['No','No','King','Japanese','Male','Code Geass','Lelouch Lamperouge'],
    ['No','No','Superhero','American','Male','First Avenger','Captain America'],
    ['No','No','Superhero','American','Male','Dark Knight','Batman'],
    ['No','No','Superhero','American','Male','Man of Steel','Superman'],
    ['No','No','Superhero','American','Female','Dawn of Justice','Wonder Woman'],
    ['No','No','Superhero','American','Female','Justice League','Jean Grey'],
    ['No','No','Superhero','American','Female','Batman','Batgirl'],
    ['No','No','Superhero','American','Female','Ant Man','Wasp'],
    ['Yes','Yes','Actor','English','Female','Atonement','Keira Knightley'],
    ['Yes','Yes','Actor','American','Female','One Day','Anne Hathaway'],
    ['Yes','No','Actor','American','Female','Divergent','Shailene Woodley'],
    ['Yes','No','Actor','American','Female','Percy Jackson','Alexandra Daddario'],
    ['Yes','No','Actor','American','Female','Vampire Academy','Zoey Deutch']
]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))

