import { NextResponse } from "next/server";
import data from "../../../util/data.json";

export const GET = async () => {
  try {
    return NextResponse.json(data, { status: 200 });
  } catch {
    return NextResponse.json(
      { message: "Failed to fetch data", success: false },
      { status: 500 }
    );
  }
};