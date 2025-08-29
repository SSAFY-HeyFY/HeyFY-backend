package com.ssafy.ssashinsa.heyfy.register.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.register.dto.AccountCreationResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "Create Foreign Currency Deposit Account", description = "Creates a new foreign currency deposit account using the Shinhan Bank API.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "Foreign currency account successfully registered.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountCreationResponseDto.class),
                        examples = @ExampleObject(
                                name = "Success Response Example",
                                value = "{\"message\": \"Processed normally.\", \"accountNo\": \"0019290964871122\", \"currency\": \"USD\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "Bad Request (e.g., user already has an account)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "User Already Has Account",
                                value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"ACCOUNT_ALREADY_EXISTS\", \"message\": \"User already has an account.\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "401",
                description = "Authentication failed (e.g., missing user key)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "Missing User Key",
                                value = "{\"status\": 401, \"httpError\": \"UNAUTHORIZED\", \"errorCode\": \"MISSING_USER_KEY\", \"message\": \"User key is missing.\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "Internal Server Error (e.g., Shinhan API call failed)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "API Call Failed",
                                value = "{\"status\": 500, \"httpError\": \"INTERNAL_SERVER_ERROR\", \"errorCode\": \"API_CALL_FAILED\", \"message\": \"Shinhan API call failed.\"}"
                        )
                )
        )
})
public @interface CreateForeignDepositAccountDocs {
}