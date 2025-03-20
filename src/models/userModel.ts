import mongoose from "mongoose";

const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, "Please provide an username"],
  },
  company: {
    type: String,
    required: [true, "Please provide a company"],
  },
  email: {
    type: String,
    required: [true, "Please provide an email"],
    unique: [true, "Please provide a unique email"],
  },
  username: {
    type: String,
    required: [true, "Please provide a username"],
  },
  password: {
    type: String,
    required: [true, "Please provide a password"],
  },
  website: {
    type: String,
    required: [true, "Please provide a website"],
  },
  category: {
    type: String,
    required: [true, "Please provide a category"],
    enum: ["Technology", "Finance", "Healthcare", "Education", "Other"],
  },
  netfunding: {
    type: Number,
    required: [true, "Please provide net funding in rupees"],
  },
mentor: {
    type: Boolean,
    required: [true, "Please specify if the user is a mentor"],
}
});

const User = mongoose.models.users || mongoose.model("users", userSchema);

export default User;
