import { NextRequest, NextResponse } from "next/server";
import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
connect();

export const GET = async (request: NextRequest) => {
  try {
    const { searchParams } = new URL(request.url);
    const username = searchParams.get("username");
    const user = await User.findOne({ username });

    if (!user) {
      return NextResponse.json(
        {
          message: "User not found",
          success: false,
        },
        { status: 404 }
      );
    }

    return NextResponse.json(
      {
        user,
        success: true,
      },
      { status: 200 }
    );
  } catch (error: any) {
    console.log(error.message);
    return NextResponse.json(
      {
        message: error.message,
        success: false,
      },
      { status: 500 }
    );
  }
};
