import NIO
import NIOHTTP1
import Foundation

class HTTPServer {
    private let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
    private var channel: Channel?

    func run(port: Int) throws {
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.configureHTTPServerPipeline().flatMap {
                    channel.pipeline.addHandler(HTTPHandler())
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 1)
            .childChannelOption(ChannelOptions.recvAllocator, value: AdaptiveRecvByteBufferAllocator())

        self.channel = try bootstrap.bind(host: "localhost", port: port).wait()
        print("Server running on localhost:8080")
    }

    func stop() {
        do {
            try self.channel?.close().wait()
            try self.group.syncShutdownGracefully()
            print("Server stopped")
        } catch {
            print("Error stopping server: \(error)")
        }
    }
}

final class HTTPHandler: ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart
    var queryParameters: [String: String] = [:]

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let reqPart = self.unwrapInboundIn(data)

        switch reqPart {
        case .head(let request):
            switch request.method {
            case .GET:
                guard let components = URLComponents(string: request.uri), components.path == "/q" else {
                        respond(context, status: .notFound, message: "Endpoint not found")
                        return
                    }
                self.queryParameters = components.queryItems?.reduce(into: [String: String]()) { result, item in
                        let replacedPlus = item.value?.replacingOccurrences(of: "+", with: " ") ?? ""
                        let decodedValue = replacedPlus.removingPercentEncoding ?? ""
                        result[item.name] = decodedValue
                    } ?? [:]

                processRequest(context: context)
            case .POST:
                // Handle POST separately in .body case
                break
            default:
                respond(context, status: .notImplemented, message: "Method not supported")
            }
        case .body(let buffer):
            if let bodyString = buffer.getString(at: 0, length: buffer.readableBytes) {
                // Assuming the body is URL-encoded form data
                self.queryParameters = parseFormURLEncodedData(data: bodyString)
            }
            processRequest(context: context)
        case .end:
            break
        }
    }

    private func processRequest(context: ChannelHandlerContext) {

        
        if !serverReady || busy {
            let reply = ["error": "busy"]
            let response = try! JSONEncoder().encode(reply)
            let responseString = String(data: response, encoding: .utf8) ?? "{}"
            respond(context, status: .ok, message: responseString)
            return
        }

        
        guard let effortString = queryParameters["effort"],
              let effort = Int(effortString) else {
              respond(context, status: .badRequest, message: "Missing or invalid parameters. Needs query=string; effort: 0-100. Optional: numTokens")
              return
        }
        
        if let tokenIdsStr = queryParameters["tokids"] {
            let tokenIds : [Int] = tokenIdsStr.split(separator: ",").compactMap { Int($0) }
            let reply = runNetwork(isTest: false, tokens: t.embed(tokenIds), effort: Double(effort)/100, srcTokenIds: tokenIds)
            let response = try! JSONEncoder().encode(reply.hitMiss)
            let responseString = String(data: response, encoding: .utf8) ?? "{}"
            respond(context, status: .ok, message: responseString)

        } else {
            let numTokens = Int(queryParameters["numtokens"] ?? "") ?? 100
            
            guard let query = queryParameters["query"],
                  let effortString = queryParameters["effort"],
                  let effort = Int(effortString) else {
                respond(context, status: .badRequest, message: "Missing or invalid parameters. Needs query=string; effort: 0-100. Optional: numTokens")
                return
            }
            
            // numTokens is optional, default to 100 if not provided or invalid
            
            // Assuming you have a function that does something with these parameters
            let result = appFunction(query: query, effort: effort, numTokens: numTokens)
            let response = try! JSONEncoder().encode(result)
            let responseString = String(data: response, encoding: .utf8) ?? "{}"
            respond(context, status: .ok, message: responseString)
        }
    }

    private func respond(_ context: ChannelHandlerContext, status: HTTPResponseStatus, message: String) {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json; charset=utf-8")
        headers.add(name: "Connection", value: "close") // Instruct the client to close the connection after receiving the response

        let response = HTTPServerResponsePart.head(HTTPResponseHead(version: .http1_1, status: status, headers: headers))
        context.write(self.wrapOutboundOut(response), promise: nil)

        let body = HTTPServerResponsePart.body(.byteBuffer(ByteBufferAllocator().buffer(string: message)))
        context.write(self.wrapOutboundOut(body), promise: nil)
        context.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        context.close(promise: nil)
    }

    private func parseFormURLEncodedData(data: String) -> [String: String] {
        return data.split(separator: "&").reduce(into: [String: String]()) { result, item in
            let components = item.split(separator: "=").map { String($0) }
            if components.count == 2 {
                result[components[0]] = components[1].removingPercentEncoding
            }
        }
    }
}

struct RequestData: Decodable {
    let query: String
}

var busy = false

func appFunction(query: String, effort: Int, numTokens: Int) -> [String: String] {
    if !serverReady {
        return ["error": "still loading weights"]
        
    }
    if busy {
        return ["error": "busy"]
    }
    busy = true
    let s = "[INST]\(query)[/INST]"
    let reply = runNetwork(isTest: false, tokens: t.embed(s), effort: Double(effort)/100, srcTokenIds: encode(prompt: s))
//    sleep(10)
    busy = false
    
    return ["response": reply.reply]

}
