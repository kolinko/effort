import NIO
import NIOHTTP1
import Foundation

class HTTPServer {
    private let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
    private var host: String
    private var port: Int

    init(host: String, port: Int) {
        self.host = host
        self.port = port
    }

    func run() throws {
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

        let channel = try bootstrap.bind(host: self.host, port: self.port).wait()
        print("Server started at \(self.host):\(self.port)")
        try channel.closeFuture.wait()
    }

    func stop() {
        do {
            try self.group.syncShutdownGracefully()
        } catch {
            print("Error shutting down server: \(error.localizedDescription)")
        }
    }
}

final class HTTPHandler: ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let reqPart = self.unwrapInboundIn(data)

        switch reqPart {
        case .head(let request):
            switch (request.method, request.uri) {
            case (.POST, "/"):
                guard request.headers["Content-Type"].contains("application/json") else {
                    self.respond(context, status: .notImplemented, message: "Content-Type not supported")
                    return
                }
            case (.GET, "/"):
                self.respond(context, status: .ok, message: "Server status: running")
                return
            default:
                self.respond(context, status: .notImplemented, message: "Not Implemented")
                return
            }
        case .body(let buffer):
            let bodyString = buffer.getString(at: 0, length: buffer.readableBytes) ?? ""
            // Here, parse the JSON and handle the query
            handlePostRequest(context: context, requestBody: bodyString)
        case .end:
            break
        }
    }

    private func handlePostRequest(context: ChannelHandlerContext, requestBody: String) {
        // Assuming requestBody contains JSON like {"query": "your_query"}
        // Parse and handle the query, then respond
        do {
            let data = Data(requestBody.utf8)
            let requestData = try JSONDecoder().decode(RequestData.self, from: data)
            let responseData = appFunction(query: requestData.query) // Your app function handling the query
            let json = try JSONEncoder().encode(responseData)
            let jsonString = String(data: json, encoding: .utf8)!

            self.respond(context, status: .ok, message: jsonString)
        } catch {
            self.respond(context, status: .badRequest, message: "Invalid request data")
        }
    }

    private func respond(_ context: ChannelHandlerContext, status: HTTPResponseStatus, message: String) {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        let response = HTTPServerResponsePart.head(HTTPResponseHead(version: .http1_1, status: status, headers: headers))
        context.write(self.wrapOutboundOut(response), promise: nil)

        let body = HTTPServerResponsePart.body(.byteBuffer(ByteBufferAllocator().buffer(string: message)))
        context.write(self.wrapOutboundOut(body), promise: nil)
        context.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        context.close(promise: nil)
    }
}

struct RequestData: Decodable {
    let query: String
}

func appFunction(query: String) -> [String: String] {
    // Implement your function logic here, based on the query
    return ["response": "Processed \(query)"]
}
