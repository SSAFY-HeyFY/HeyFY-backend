package com.ssafy.ssashinsa.heyfy.transfer.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;
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
@Operation(
        summary = "원화 계좌 이체",
        description = "사용자의 원화 출금 계좌에서 지정된 입금 계좌로 금액을 이체합니다."
)
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 이체를 완료했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = TransferResponseDto.class),
                        examples = @ExampleObject(
                                name = "원화 이체 성공",
                                value = "{\n  \"depositAccountNo\": \"110123456789\",\n  \"amount\": \"10000\",\n  \"currency\": \"KRW\",\n  \"transactionSummary\": \"점심값\",\n  \"completedAt\": \"2025-08-26T23:50:15.123456+09:00\"\n}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "PIN 번호 불일치",
                                        summary = "PIN 번호 오류",
                                        value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"INVALID_PIN_NUMBER\", \"message\": \"Wrong pin number\"}"
                                ),
                                @ExampleObject(
                                        name = "금액 오류",
                                        summary = "입력한 금액이 유효하지 않음",
                                        value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"INVALID_TRANSACTION_AMOUNT\", \"message\": \"The transaction amount is invalid\"}"
                                ),
                                @ExampleObject(
                                        name = "잔액 부족",
                                        summary = "잔액 부족",
                                        value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"INSUFFICIENT_BALANCE\", \"message\": \"Transaction failed due to insufficient account balance\"}"
                                ),
                                @ExampleObject(
                                        name = "계좌번호 오류",
                                        summary = "계좌번호 유효성 오류",
                                        value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"INVALID_ACCOUNT_NUMBER\", \"message\": \"The account number is invalid\"}"
                                ),
                                @ExampleObject(
                                        name = "거래 내용 길이 초과",
                                        summary = "거래 내용 길이 초과",
                                        value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"EXCEEDED_TRANSACTION_SUMMARY_LENGTH\", \"message\": \"The transaction summary length has been exceeded\"}"
                                ),
                                @ExampleObject(
                                        name = "유저 키 누락",
                                        summary = "인증 정보에서 유저 키를 찾을 수 없음",
                                        value = "{\"status\": 400, \"httpError\": \"UNAUTHORIZED\", \"errorCode\": \"MISSING_USER_KEY\", \"message\": \"유저키가 누락되었습니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "유저 없음",
                                        summary = "인증 정보에서 유저를 찾을 수 없음",
                                        value = "{\"status\": 400, \"httpError\": \"UNAUTHORIZED\", \"errorCode\": \"USER_NOT_FOUND\", \"message\": \"유저를 찾을 수 없습니다.\"}"
                                ),
                        }
                )
        ),
//        @ApiResponse(
//                responseCode = "400",
//                description = "인증 실패",
//                content = @Content(
//                        mediaType = "application/json",
//                        schema = @Schema(implementation = ErrorResponse.class),
//                        examples = {
//                                @ExampleObject(
//                                name = "유저 키 누락",
//                                summary = "인증 정보에서 유저 키를 찾을 수 없음",
//                                value = "{\"status\": 400, \"httpError\": \"UNAUTHORIZED\", \"errorCode\": \"MISSING_USER_KEY\", \"message\": \"유저키가 누락되었습니다.\"}"
//                                ),
//                                @ExampleObject(
//                                        name = "유저 없음",
//                                        summary = "인증 정보에서 유저를 찾을 수 없음",
//                                        value = "{\"status\": 400, \"httpError\": \"UNAUTHORIZED\", \"errorCode\": \"USER_NOT_FOUND\", \"message\": \"유저를 찾을 수 없습니다.\"}"
//                                ),
//                        }
//                )
//        )
})
@Parameter(name = "Authorization", description = "JWT 액세스 토큰 (Bearer <token>)", in = ParameterIn.HEADER, required = true)
@Parameter(name = "TxnAuthToken", description = "2차 인증을 위한 트랜잭션 토큰", in = ParameterIn.HEADER, required = true)
@Parameter(name = "sid", description = "세션 ID", in = ParameterIn.HEADER, required = true)
public @interface TransferDocs {
}