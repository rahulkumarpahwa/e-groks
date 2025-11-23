import { NextResponse } from "next/server";

export const GET = async () => {

    return NextResponse.json(
        {
            message: "backend is up!",
            success: true,
            usage: "Use any of the routes listed above with GET method",
            examples: {
                getGraphicsdoth: "GET /api/graphics",
                getlibbgidota: "GET /api/libbgi",
                getwinbgim: "GET /api/winbgim",
                getCommand: "GET /api/readme",
                getc: "GET /api/c",
                getcdata : "GET /api/cdata"
            }
        },
        { status: 200 }
    );
};