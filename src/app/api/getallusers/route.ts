import { NextRequest, NextResponse } from "next/server";
import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
connect();

export const GET = async (request: NextRequest) => {
  try {
    const users = await User.find();

    if (!users || users.length === 0) {
      return NextResponse.json(
        {
          message: "No users found",
          success: false,
        },
        { status: 404 }
      );
    }

    return NextResponse.json(
      {
        users,
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
