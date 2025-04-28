import { NextResponse } from "next/server";
export const GET = async () => {
    return NextResponse.json(
        {
            message: "backend is up!",
            success: true,
        },
        { status: 200 }
    );
};