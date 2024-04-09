//
//  tokeniser2.swift
//  mul_col
//
//  Ported from the Javascript implementation by belladore.ai
//  github.com/imoneoi/mistral-tokenizer/blob/master/mistral-tokenizer.js
//
//

import Foundation

/**
 * Helper function to decode the vocabulary.
 *
 * vocab_base64 is base64-encoded string of tokens delimited by '\n' (line break) in utf-8.
 * The row number of the token (indexing from 0) represents the id of the token in mistral tokenizer.
 *
 * Most tokens look like this: "ic" (without the quotes) (representing the "i" character followed by the "c" character)
 * Some tokens are special. In particular, spaces are replaced with the "▁" character and line-break is represented as "<0x0A>".
 *
 * This helper function returns the vocabulary as an array that contains Strings representing tokens:
 *
 *  "<unk>"   // Special token: unknown token
 *  "<s>"     // Special token: beginning of string
 *  "</s>"    // Special token: end of string
 *  "<0x00>"  // Byte-level token representing the 0-byte
 *  "<0x01>"  // Byte-level token ...
 *  "<0x02>"  // Byte-level token ...
 *  ...       // More byte-level tokens
 *  "<0x0A>"  // Byte-level token representing '\n' (line break). This is one of the few byte-level tokens that appear to be actually needed in practice.
 *  ...       // More byte-level tokens
 *  "<0xFF>"  // Byte-level token ...
 *  "▁▁"     // Token representing 2 consecutive spaces.
 *  "▁t"     // Token representing the space character followed by the "t" character.
 *  "er"      // Token representing the "e" character followed by the "r" character. Most tokens look like this.
 *  ...       // 32000 tokens
 */


// Assuming the JSON structure based on your Python code
private struct ModelData: Decodable {
    let vocab: [String:Int]
    let merges: [String]
}

private struct TokenizerData: Decodable {
    let model: ModelData
}

class MistralTokenizer {
    var vocabById: [Int: String]
    var vocabByString: [String: Int]
    var merges: [String: Int]
    var spaceToken: String
    init() {
        print("init!")
        // Load and decode JSON
        let fileUrl = URL(fileURLWithPath: absolutePath + "tokenizer.json")
        let data = try! Data(contentsOf: fileUrl)
        let decoder = JSONDecoder()
        let decodedData = try! decoder.decode(TokenizerData.self, from: data)
        
        // Extract vocab and merges directly
        let vocab = decodedData.model.vocab
        let merges = decodedData.model.merges
        
        // Create vocabById mapping
        self.vocabById = [Int: String]()
        
        // Create vocabByString mapping
        self.vocabByString = vocab
        for key in self.vocabByString.keys {
            self.vocabById[self.vocabByString[key]!] = key
        }
        
        // Process merges into binary format (skipped for brevity)
        // Here, you'd transform 'merges' into your desired binary representation
        var i = 0;
        self.merges = [String: Int]()
        for merge in merges {
            /*
            let parts = merge.split(separator: " ").map(String.init)
            let id1 = vocabByString[parts[0]]!
            let id2 = vocabByString[parts[1]]!
            let mergeIdentifierString = "\(id1) \(id2)"*/
            // Key identifies token pair, value represents merge priority
            self.merges[merge] = i+1
            i+=1
        }
        
        spaceToken = vocabById[28705]!
        print("done!")
    }
}

private let mistralTokenizer = MistralTokenizer()

private func getMergeIdentifierString(_ firstTokenId: Int, _ secondTokenId: Int) -> String {
    return mistralTokenizer.vocabById[firstTokenId]! + " " + mistralTokenizer.vocabById[secondTokenId]!
}

// Helper function to convert a UTF-8 byte to a hex string
private func utf8ByteToHex(_ c: UInt8) -> String {
    return String(format: "<0x%02X>", c)
}

// Helper function to convert a hex string back to a UTF-8 byte
private func hexToUtf8Byte(_ hex: String) -> UInt8? {
    let strippedHex = hex.trimmingCharacters(in: CharacterSet(charactersIn: "<0x>"))
    return UInt8(strippedHex, radix: 16)
}

private func mapCharactersToTokenIds(prompt _prompt: String, addBosToken: Bool, addPrecedingSpace: Bool) -> [Int] {
    var tokenIds = [Int]()
    if addBosToken {
        tokenIds.append(1)
    }
    
    var prompt = _prompt
    
    if addPrecedingSpace {
        prompt = " " + prompt
    }

    // Special: spaces are represented as thick underscore ▁ (id 28705)
    let promptAltered = prompt.replacingOccurrences(of:" ", with: mistralTokenizer.spaceToken)

    // Transform each character to its corresponding token
    for c in promptAltered {
//        print(".", terminator: "")

        if let tokenId = mistralTokenizer.vocabByString[String(c)] {
            tokenIds.append(tokenId)
        } else {
            let bytes = Array(c.utf8)
            // Special case where token not found and we have to fallback to byte-level tokens.
            for byte in bytes {
                let hex = utf8ByteToHex(byte)
                if let hexValue = mistralTokenizer.vocabByString[hex] {
                            tokenIds.append(hexValue)
                        } else {
                            print("Encountered unknown character \(c) (partial UTF-8 byte \(byte) + hex + \(hex))")
                             tokenIds[tokenIds.count - 1] = 0 // Assuming 0 is the <UNK> token
                        }
                }
        }
    }

    return tokenIds
}

private class Node : Comparable {
    let tokenId : Int
    var prev : Node? = nil
    var next : Node? = nil
    let origPos : Int
    var mergePrio : Double = 0
    var mergeToString : String = ""
    var deleted : Bool = false
    
    init(tokenId: Int, prev: Node? = nil, next: Node? = nil, origPos: Int, mergePrio: Double = 0, mergeToString: String = "") {
        self.tokenId = tokenId
        self.prev = prev
        self.next = next
        self.origPos = origPos
        self.mergePrio = mergePrio
        self.mergeToString = mergeToString
    }
    
    // Implementation of the Equatable protocol
    static func ==(lhs: Node, rhs: Node) -> Bool {
        return lhs.mergePrio == rhs.mergePrio
    }
    
    // Implementation of the Comparable protocol
    static func <(lhs: Node, rhs: Node) -> Bool {
        return lhs.mergePrio < rhs.mergePrio
    }
}

func encode (prompt: String, addBosToken: Bool = true, addPrecedingSpace: Bool = true) -> [Int] {
    if prompt == "" { return [Int]() }
    let tokenIds = mapCharactersToTokenIds(prompt: prompt, addBosToken:addBosToken, addPrecedingSpace: addPrecedingSpace)
    
    let mergeQueue = PriorityQueue<Node>()
    
    func addToMergeQueue(_ leftNode: Node) {

        // Merge priority is primarily determined by the location of the merge in the "merges" data,
        // secondarily determined by the relative position of the node in the linked list
        // (We want to perform equal merges from left to right)

        let mergeIdentifierString = getMergeIdentifierString(leftNode.tokenId, leftNode.next!.tokenId)

        // If mergePrio not found in merges, that means this merge is not possible according to vocabulary.
        if let mergePrio = mistralTokenizer.merges[mergeIdentifierString] {
            leftNode.mergePrio = Double(mergePrio) + Double(leftNode.origPos) / Double(prompt.count)
            leftNode.mergeToString = mergeIdentifierString.replacingOccurrences(of: " ", with: "")
            _ = mergeQueue.push(leftNode)
        }
    }
   
    var firstTokenNode = Node(
        tokenId: tokenIds[0],
        prev: nil,
        next: nil,
        origPos: 0)

    var prevTokenNode = firstTokenNode
    
    for i in 1..<tokenIds.count {
        let currTokenNode = Node(
            tokenId: tokenIds[i],
            prev: prevTokenNode,
            next: nil,
            origPos: i)
        
        prevTokenNode.next = currTokenNode
        addToMergeQueue(prevTokenNode)
        prevTokenNode = currTokenNode

    }

    // Perform merges in priority order
    while (!mergeQueue.isEmpty) {
        let leftOfMerge = mergeQueue.pop()

        // Check that this merge is still possible
        if (leftOfMerge.deleted) { continue }
        if (leftOfMerge.next == nil) { continue }
        if (leftOfMerge.next!.deleted) {continue }
        
        // Mark leftOfMerge and rightOfMerge as being deleted, because they are actually being replaced by a merged token.
        leftOfMerge.deleted = true
        leftOfMerge.next!.deleted = true
        
        // It's a little bit more complicated to fix the prev of leftOfMerge.
        if (leftOfMerge.prev != nil) {
            let oldPrev = leftOfMerge.prev!
            // Mark oldPrev as deleted, to avoid erroneous merges later (ref to this node might exist in priorityqueue)
            oldPrev.deleted = true
            // Replace oldPrev within the linked list with a copy of itself
            let newPrev = Node(
                tokenId: oldPrev.tokenId,
                prev: oldPrev.prev,
                next: oldPrev.next,
                origPos: oldPrev.origPos)
            
            leftOfMerge.prev = newPrev
            // Update linked list reference of "prev of prev"
            if (newPrev.prev != nil) {
                newPrev.prev!.next = newPrev
            } else {
                // If "prev of prev" does not exist, that means newPrev must be the new firstNode
                firstTokenNode = newPrev
            }
        }
        // Create node representing merge result
        let resultOfMerge = Node(
            tokenId: mistralTokenizer.vocabByString[leftOfMerge.mergeToString]!,
            prev: leftOfMerge.prev,
            next: leftOfMerge.next!.next,
            origPos: leftOfMerge.origPos)
            
        // Consider adding to merge queue: prev--resultOfMerge
        if (resultOfMerge.prev != nil) {
            resultOfMerge.prev!.next = resultOfMerge
//            resultOfMerge.prev
            addToMergeQueue(resultOfMerge.prev!)
        } else {
            // If prev does not exist then this is the new firstNode
            firstTokenNode = resultOfMerge
        }
        // Consider adding to merge queue: resultOfMerge--next
        if (resultOfMerge.next != nil) {
            resultOfMerge.next!.prev = resultOfMerge
            addToMergeQueue(resultOfMerge)
        }

    }

    // Get final tokenIds by traversing the linked list
    var mergedTokenIds = [Int]()
    var currTokenNode: Node? = firstTokenNode
    
    while let node = currTokenNode {
        mergedTokenIds.append(node.tokenId)
        currTokenNode = node.next
    }

    return mergedTokenIds
}


class PriorityQueue<Type: Comparable> {
    var _heap = [Type]()
    // PriorityQueue implementation is copied from stackoverflow.com/a/42919752 with minor refactoring
    // then ported to Swift and replaced with a list, because I'm lazy (TK)
    
    var size : Int { _heap.count }
    var bottom : Int { _heap.count-1 }

    var isEmpty : Bool { _heap.count == 0}
    
    func peek() -> Type {
        return _heap[0]
    }
    
    func push(_ el: Type) -> Int {
        _heap.append(el)
        _siftUp()
        return self.size
    }
    
    func push(_ values: [Type]) -> Int {
        for el in values {
            _ = push(el)
        }
        return self.size
    }
    
    func pop() -> Type {
        let poppedValue = peek()
        if bottom > 0 {
            _swap(0, bottom)
        }
        _heap.removeLast()
        _siftDown()
        return poppedValue
    }
    
    func replace(_ value: Type) -> Type {
        let replacedValue = peek()
        _heap[0] = value;
        _siftDown()
        return replacedValue
    }
    
    func _swap(_ a: Int, _ b: Int) {
        let tmp = _heap[a]
        _heap[a] = _heap[b]
        _heap[b] = tmp
    }
    
    func _siftUp() {
        for i in 0..<size-1 {
            if _heap[i] > _heap[i+1] {
                _swap(i, i+1)
            }
        }
    }
    
    func _siftDown() {
        if size > 1 {
            for i in (0..<size-1).reversed() {
                if _heap[i] > _heap[i+1] {
                    _swap(i, i+1)
                }
            }
        }
    }
}



func runTokenizerTests() {

    func testCase(_ inputString: String, _ expectedTokenIds: [Int]) {
        let actualTokens = encode(prompt: inputString, addBosToken:true, addPrecedingSpace: true)
        assert(actualTokens == expectedTokenIds, "Test failed. mistral Tokenizer Encoder returned unexpected result: expected tokenize('\(inputString)') === \(expectedTokenIds), actual was: \(actualTokens)")

        /*        if (inputString !== decode(actualTokens)) {
            throw `Test failed. mistral Tokenizer Decoder returned unexpected result: expected decode(${actualTokens}) === ${inputString}, actual was: ${decode(actualTokens)}`
        }*/
    }
        
    // Simple test case
    testCase("grabbed",                           [1, 13300])

    // Naive implementation produces inconsistent tokenization for " grabbed", making this a good test case
    testCase(" grabbed",                          [1, 28705, 13300])

    // Naive implementation uses incorrect merge order for multiple consecutive space merges, making this a good test case
    testCase("           grabbed",                [1, 17422, 13300])

    // Linebreaks and tabs are handled as fallback to byte tokens
    testCase("\n",                                [1, 28705, 13])
    testCase(" \n",                               [1, 259,   13])
    testCase("\ttabs\t\t\t\tout here",    [1, 28705, 12, 24856, 12, 12, 12, 12, 406, 1236])

    // Equal prio merges are performed left-to-right (fixed in 1.1.1)
    testCase("ax\n####\nboo",                     [1, 6056, 13, 2000, 13, 1798, 28709])

    // UTF-8 multipoint character that should be found in vocabulary
    testCase("镇",                                [1, 28705, 29780])

    // UTF-8 multipoint character that should NOT be found in vocabulary, fallback to MULTIPLE byte tokens
    testCase("🦙",                               [1, 28705, 243, 162, 169, 156])

    // Consecutive UTF-8 multipoint characters that are NOT found in a vocabulary and use DIFFERENT number of bytes
    testCase("🦙Ꙋ",                              [1, 28705, 243, 162, 169, 156, 237, 156, 141])
    testCase("Ꙋ🦙",                              [1, 28705, 237, 156, 141, 243, 162, 169, 156])

    // Larger text input with various special characters sprinkled in
    /*
    testCase("The llama (/ˈlɑːmə/; 🦙Spanish pronunciation: [ˈʎama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era. Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled \"lama\" or \"glama\") was adopted by European settlers from native Peruvians.[4] The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000Ꙋ🦙 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5] In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and lagoons where they come from at the end of time.[6]",
    [1, 415, 8814, 2786, 325, 28748, 29097, 28714, 29813, 29240, 28719, 29108, 28748, 28745, 28705, 243, 162, 169, 156, 13116, 789, 12704, 14281, 352, 28747, 733, 29097, 205, 145, 2786, 2803, 325, 28758, 2786, 1272, 2786, 28731, 349, 264, 2853, 374, 6899, 3658, 2556, 3730, 301, 313, 28725, 12575, 1307, 390, 264, 10228, 304, 2163, 8527, 486, 1015, 28706, 276, 19826, 1854, 272, 4258, 28733, 1577, 2915, 753, 4204, 28723, 393, 5989, 293, 460, 2809, 8222, 304, 2943, 395, 2663, 390, 264, 559, 28715, 28723, 6723, 25943, 349, 2664, 304, 5876, 865, 264, 1741, 3558, 302, 26573, 27545, 20011, 28750, 28793, 393, 5989, 293, 541, 2822, 3588, 9796, 1024, 264, 1664, 21435, 2065, 28723, 1684, 1413, 264, 2163, 28725, 590, 541, 7096, 684, 28705, 28750, 28782, 298, 28705, 28770, 28734, 28823, 302, 652, 2187, 4336, 354, 28705, 28783, 298, 28705, 28740, 28770, 3535, 325, 28782, 28816, 28783, 6052, 609, 28792, 28770, 28793, 415, 1141, 8814, 2786, 325, 262, 272, 2609, 835, 668, 6099, 345, 28714, 2786, 28739, 442, 345, 1727, 2786, 1243, 403, 13424, 486, 6392, 4641, 8531, 477, 8271, 2744, 5388, 3693, 20011, 28781, 28793, 415, 25427, 302, 17620, 293, 460, 1654, 298, 506, 5016, 601, 477, 272, 6043, 1641, 1606, 302, 3964, 4352, 684, 28705, 28781, 28734, 3841, 1267, 3584, 28725, 304, 18410, 11205, 601, 298, 3658, 4352, 684, 1712, 3841, 1267, 3584, 1938, 272, 6043, 2556, 4287, 4078, 28723, 2463, 272, 948, 302, 272, 1432, 7515, 3595, 325, 28740, 28734, 28725, 28734, 28734, 28734, 28816, 28740, 28750, 28725, 28734, 28734, 28734, 1267, 3584, 557, 3730, 301, 2298, 654, 1568, 5654, 297, 3964, 4352, 20011, 28770, 28793, 1136, 302, 28705, 28750, 28734, 28734, 28787, 28725, 736, 654, 754, 6671, 3841, 17620, 293, 304, 389, 28720, 323, 293, 297, 3658, 4352, 304, 754, 28705, 28740, 28782, 28783, 28725, 28734, 28734, 28734, 17620, 293, 304, 28705, 28740, 28734, 28734, 28725, 28734, 28734, 28734, 237, 156, 141, 243, 162, 169, 156, 389, 28720, 323, 293, 28725, 2283, 2508, 477, 430, 2383, 9058, 26659, 3909, 297, 272, 28705, 28750, 28734, 362, 5445, 28725, 297, 272, 2969, 3543, 304, 6082, 20011, 28782, 28793, 560, 330, 1082, 2923, 13345, 2161, 28725, 17620, 293, 460, 2278, 16905, 28723, 415, 22830, 346, 393, 28714, 2786, 349, 773, 298, 4663, 2130, 477, 272, 13993, 304, 4273, 10218, 390, 378, 408, 1606, 20011, 28784, 28793, 6586, 298, 330, 1082, 2923, 1037, 12932, 2161, 28725, 17620, 293, 622, 604, 298, 272, 2130, 7474, 28713, 304, 305, 4567, 1053, 970, 590, 1567, 477, 438, 272, 948, 302, 727, 20011, 28784, 28793])*/

    print("Mistral Tokenizer tests passed successfully.")
}
