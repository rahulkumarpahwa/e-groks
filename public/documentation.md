## API Documentation

### 1. Initialize Users Data

**Endpoint:** `/api/init`

**Method:** `POST`

**Description:** This endpoint initializes the users data by inserting 10 dummy users from the `data.ts` file into the database.

**Request Body:** None

**Response:**

- **201 Created:** Successfully inserted users.
  ```json
  {
    "newUsers": [
      /* array of user objects */
    ],
    "success": true
  }
  ```
- **500 Internal Server Error:** An error occurred while inserting users.
  ```json
  {
    "message": "Error message",
    "success": false
  }
  ```

### 2. Get All Users

**Endpoint:** `/api/getallusers`

**Method:** `GET`

**Description:** This endpoint retrieves all users from the database.

**Request Parameters:** None

**Response:**

- **200 OK:** Successfully retrieved users.
  ```json
  {
    "users": [
      /* array of user objects */
    ],
    "success": true
  }
  ```
- **404 Not Found:** No users found.
  ```json
  {
    "message": "No users found",
    "success": false
  }
  ```
- **500 Internal Server Error:** An error occurred while retrieving users.
  ```json
  {
    "message": "Error message",
    "success": false
  }
  ```

### 3. Get User Details by Username

**Endpoint:** `/api/getuserdetails`

**Method:** `GET`

**Description:** This endpoint retrieves user details based on the username.

**Request Parameters:**

- `username` (string): The username of the user to retrieve.

**Response:**

- **200 OK:** Successfully retrieved user details.
  ```json
  {
    "user": {
      /* user object */
    },
    "success": true
  }
  ```
- **404 Not Found:** User not found.
  ```json
  {
    "message": "User not found",
    "success": false
  }
  ```
- **500 Internal Server Error:** An error occurred while retrieving user details.
  ```json
  {
    "message": "Error message",
    "success": false
  }
  ```

### 4. Get Mentors by Category

**Endpoint:** `/api/getmentorscategory`

**Method:** `GET`

**Description:** This endpoint retrieves mentors based on the category.

**Request Parameters:**

- `category` (string): The category to filter mentors by.

**Response:**

- **200 OK:** Successfully retrieved mentors.
  ```json
  {
    "users": [
      /* array of user objects */
    ],
    "success": true
  }
  ```
- **404 Not Found:** No mentors found in this category.
  ```json
  {
    "message": "No mentors found in this category",
    "success": false
  }
  ```
- **500 Internal Server Error:** An error occurred while retrieving mentors.
  ```json
  {
    "message": "Error message",
    "success": false
  }
  ```

## Data Model

### User Model

**File:** `src/models/userModel.ts`

**Schema:**

```typescript
import mongoose from "mongoose";

const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, "Please provide a name"],
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
  },
  image: {
    type: String,
    default:
      "https://res.cloudinary.com/dwtcjjxwc/image/upload/v1742500586/user_h2ggtz.png",
  },
});

const User = mongoose.models.users || mongoose.model("users", userSchema);

export default User;
```

## Database Configuration

**File:** `src/dbConfig/dbConfig.ts`

**Configuration:**

```typescript
import mongoose from "mongoose";

export const connect = async () => {
  try {
    mongoose.connect(process.env.MONGO_URL!, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    // Event listeners
    mongoose.connection.on("connected", () => {
      console.log("Database is connected");
    });
    mongoose.connection.on("error", (err) => {
      console.log("MongoDB connection error: " + err);
      process.exit(); // Exit the code and not run the app further without database.
    });
  } catch (error) {
    console.log("Something went wrong in connecting to the database");
    console.log(error);
  }
};
```

### Environment Variables

**File:** `.env`

```properties
MONGO_URL="mongodb://127.0.0.1:27017/grok"
```

This documentation provides an overview of the available APIs, the user data model, and the database configuration for your project.
