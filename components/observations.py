from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
import numpy as np


class CustomObservation(TreeObsForRailEnv):
    def __init__(self, max_depth=2, predictor=ShortestPathPredictorForRailEnv(30)):
        super().__init__(max_depth=max_depth, predictor=predictor)

    def get(self, handle: int = 0):
        obs = super(CustomObservation, self).get(handle=handle)
        if obs:
            obs = self.tree_encoding(obs)
        return obs

    '''DECOMPOSING PART'''

    '''tree_decomposition(obs) is used to decompose the observation of a single agent into a list of all the nodes that 
    compose the tree observation '''

    def tree_decomposition(self, obs):
        """
        tree_decomposition(obs) is used to decompose the observation of a single agent into a list of all the
        nodes that compose the tree observation
        :param obs: is the observation of a single agent
        :return:
        """
        tree_list = [self.node_decomposition(obs)]
        childs = self.childs_decomposition(obs[12], 1)
        # call for the child_decomposition(obs[12], 1) function: obs[12] is the thirteenth attribute of a Node object;
        # '1' as second argument is used to initialize the depth parameter for gathering information about nodes
        # (see the below exaplanation of the function).

        tree_list.append(childs)

        return tree_list

    '''node_decomposition(node) is used to separate and store into a single value all the information about a node.'''

    @staticmethod
    def node_decomposition(node):
        data = np.zeros(12)

        data[0] = node[0]  # information acquisition
        data[1] = node[1]
        data[2] = node[2]
        data[3] = node[3]
        data[4] = node[4]
        data[5] = node[5]
        data[6] = node[6]
        data[7] = node[7]
        data[8] = node[8]
        data[9] = node[9]
        data[10] = node[10]
        data[11] = node[11]

        for i in range(len(
                data)):  # here 'inf' value are converted into '-1', at least at the beginning in order to be then added
            # together

            if data[i] == np.inf:
                data[i] = -1

        data[0] = data[0] * 1  # here, weights for each characteristic are given in order to give more importance to
        # certain elements during the sum
        data[1] = data[1] * 0.3
        data[2] = data[2] * 0.6
        data[3] = data[3] * 0.8
        data[4] = data[4] * 0.3
        data[5] = data[5] * 0.3
        data[6] = data[6] * 1
        data[7] = data[7] * 0.6
        data[8] = data[8] * 0.8
        data[9] = data[9] * 0.3
        data[10] = data[10] * 0.3
        data[11] = data[11] * 0.1

        return np.sum(data)

    '''childs_decomposition(childs, depth) is used to get a list of all the children of a node'''

    def childs_decomposition(self, childs, depth):
        childs_list = []

        for direction in childs:  # since in 'childs' is a dict

            if childs[direction] == -np.inf:

                minus_one_array = - np.ones(
                    12)  # '-np.inf'is substituted with an array of length 12, whose elements are -1. In this way,
                # we have observations of the same length.

                if depth < self.max_depth:  # here, the part of filling the observation starts. This must be done in
                    # order to have observations of the same dimension

                    fill_childs_list = []  # this is the final list that will be used to add all the missing nodes (when
                    # we have reached the maximum depth)

                    letters_list = ['L', 'F', 'R', 'B']  # a list that I need to iterate

                    non_existing_childs = self.fill_childs(fill_childs_list, depth,
                                                           letters_list)  # call for the fill_childs() function (see
                    # explanation below)

                    childs_list.append((direction, depth, np.sum(minus_one_array),
                                        non_existing_childs))  # here, all the components are added together

                else:  # this is referred to the case in which we have reached the maximum depth (in fact there's no
                    # fourth element because there's no need to fill the obs)

                    childs_list.append((direction, depth, np.sum(minus_one_array)))

            else:
                # this is referred to all the nodes that contain relevant information (different from -np.inf)

                node_data = self.node_decomposition(childs[direction])

                if childs[direction][12] != {}:  # if there are children

                    childs_node_data = self.childs_decomposition(childs[direction][12], depth + 1)
                    # here, childs_decomposition() function is called recursively
                    childs_list.append((direction, depth, node_data, childs_node_data))

                else:  # otherwise..it means that we have reached the end of the observation

                    childs_list.append((direction, depth, node_data))

        return childs_list

    '''fill_childs(fill_childs_list, depth, letters_list) is used to fill the remaining part of the observation in order 
    to have observations of the same dimension '''

    def fill_childs(self, fill_childs_list, depth, letters_list):
        if depth < self.max_depth:

            for letter in letters_list:  # just take a single letter

                elem = (letter, depth + 1, -1)  # generate the element to be added

                fill_childs_list.append(elem)  # add the element

                temp_list = self.fill_childs(fill_childs_list, depth + 1,
                                             letters_list)  # then I call recursively the fill_childs() function

                fill_childs_list.append(temp_list)  # and then append the result

        return fill_childs_list

    '''ENCODING AND CONCATENATION PART'''

    '''tree_encoding(obs) is used to convert the encode the value of the 4 directions and to concatenate all the values 
    in a single array of fixed length '''

    def tree_encoding(self, obs):  # obs is the observation of a single agent

        tree_decomposed = self.tree_decomposition(obs)  # call for the tree_decomposition() function

        final_array = np.array([0, 0, tree_decomposed[0]])  # initialization of the resulting array

        tree_dict = {  # dict for converting the directions in number

            'L': 1,
            'F': 2,
            'R': 3,
            'B': 4
        }

        final_array = self.sub_tree_encoding(tree_decomposed[1], final_array, tree_dict)
        # call of the sub_tree_encoding() function (see below for explanation)

        return final_array

    '''sub_tree_encoding is used to navigate the tree in order to encode directions and concatenate all the values'''

    def sub_tree_encoding(self, tree_decomposed, final_array, tree_dict):
        for sub_tree in tree_decomposed:

            if len(sub_tree) == 3:  # this is the base, that is when a node is not characterized by any child
                final_array = np.append(final_array, tree_dict[sub_tree[0]])  # append the encoded direction
                final_array = np.append(final_array, sub_tree[1])  # append the depth
                final_array = np.append(final_array, sub_tree[2])  # append the value (the sum of the array)

            elif len(sub_tree) == 4 and isinstance(sub_tree[-1], list):
                # this is the case in which a node is charaterized by a fourth
                # element that is a list of children

                final_array = np.append(final_array, tree_dict[sub_tree[0]])  # append the encoded direction
                final_array = np.append(final_array, sub_tree[1])  # append the depth
                final_array = np.append(final_array, sub_tree[2])  # append the value

                final_array = self.sub_tree_encoding(sub_tree[3], final_array, tree_dict)
                # call recurisvely the sub_tree_encoding function,
                # passing as 'tree_decomposed' element the list of children

        return final_array
