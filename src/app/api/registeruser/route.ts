import { NextRequest, NextResponse } from "next/server";
import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
connect();

export const POST = async (request: NextRequest) => {
    try {
        const reqBody = await request.json();
        const { name, company, email, username, password, website, category, netfunding, mentor } = reqBody;

        const newUser = new User({
            name,
            company,
            email,
            username,
            password,
            website,
            category,
            netfunding,
            mentor,
        });

        await newUser.save();

        return NextResponse.json(
            {
                newUser,
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

