import { NextRequest, NextResponse } from "next/server";
import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
import usersData from "./data";
connect();

export const POST = async (request: NextRequest) => {
  try {
    const users = usersData;

    const newUsers = await User.insertMany(users);

    return NextResponse.json(
      {
        newUsers,
        success: true,
      },
      { status: 201 }
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
