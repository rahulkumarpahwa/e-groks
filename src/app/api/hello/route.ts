import { NextRequest, NextResponse } from "next/server";
export const GET = async (request: NextRequest) => {
    return NextResponse.json(
        {
            message: "backend is up!",
            success: true,
        },
        { status: 200 }
    );
};